#!/usr/bin/env python3
"""
Music Tone Classifier using LAION CLAP model (aarush_base_model_v1)
This system can:
1. Classify audio files to tone effects and provide knob settings
2. Match text descriptions to appropriate tone effects and knob settings
Model: mlp
Model settings comments: This is an MLP model
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import librosa
from transformers import ClapModel, ClapProcessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import pickle
from datetime import datetime

import time
import copy
import warnings
import random
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================
# CLAP MODEL CONFIGURATION
# ============================
#CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"

# -----------------------------------------------------------------------------
# Reproducibility: fix random seeds and deterministic behaviour
# -----------------------------------------------------------------------------
SEED = 201  # pick any integer and keep it fixed for debugging

# Python's own RNG
random.seed(SEED)

# NumPy RNG
np.random.seed(SEED)

# PyTorch CPU RNG
torch.manual_seed(SEED)

# PyTorch CUDA RNG (if GPU is available)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# cuDNN deterministic behaviour (may be a bit slower, but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Pre-split total training samples (set by the driver script; read-only for report)
PRE_SPLIT_TRAIN_SAMPLES = None
# -----------------------------------------------------------------------------
# REPORT_TUNING (User-editable; used ONLY for reporting. Does NOT affect models.)
# Initial keys per request:
#   1) samples_used_for_training
#   2) datasets_used (you may write whatever you want to report; we also
#      show self.dataset_toggles during report for convenience)
#   3) batch_size
#   4) learning_rate
#   5) epochs (actual epochs where model stopped, as you want to report)
#   6) dropout_rate
#   7) user_comments (free text)
#   8) model_name  -> also used to suffix the report filename (_val_<model>)
# Fill these manually; nothing here is read by training.
# -----------------------------------------------------------------------------
REPORT_TUNING = {
    "samples_used_for_training": None,
    "datasets_used": None,      # e.g., {"temporal_aware": 1, "tempboost_aware": 0} or any string you prefer
    "batch_size": None,
    "learning_rate": None,
    "epochs": None,             # set to what you want to report
    "dropout_rate": None,
    "weight_decay": None,
    "user_comments": "mlp testing - base, no cleaning, weight_decay=1e-5, nn.LeakyReLU(0.1), learning rate scheduler: ReduceLROnPlateau, StandardScaler",
    "model_name": "mlp",        # controls filename suffix _val_<model>
}


class KnobParameterNet(nn.Module):
    """Neural Network for predicting all 17 knob parameters with shared representations"""
    
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64], output_dim=17, dropout_rate=0.25):
        super(KnobParameterNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with batch normalization and dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.GELU(),
                #nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer with sigmoid activation to constrain outputs to [0, 1]
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)



class MusicToneClassifier:
    
    # --- CENTRAL CONFIGURATION START (From Run 16) ---
    DEFAULT_CLEANING_THRESHOLDS = {
        'n_chorusmix': 0.10,   
        'n_delaymix': 0.20,    
        'n_reverbwet': 0.25,   
    }
    # --- CENTRAL CONFIGURATION END ---

    def __init__(self, data_dir="Data", audio_length=5.0): # <--- Added audio_length arg
        # === base folders ===
        self.data_dir = data_dir
        self.index_dir = os.path.join(data_dir, "indexes")
        self.audio_root = os.path.join(data_dir, "generated_audio")
        self.labels_root = os.path.join(data_dir, "labels_normalised")

        # --- AUDIO CONFIGURATION ---
        self.audio_length = audio_length
        self.target_sr = 48000
        
        # Statistics Counters (Internal)
        self._stat_padded = 0
        self._stat_cropped = 0
        self._stat_total_processed = 0

        # Dataset toggle matrix (simplified for MLP)
        self.dataset_toggles = {
            "temporal_aware": 0,
            "tempboost_aware": 0,
            "static_aware": 0,
            "clean_data": 1,  # <--- NEW TOGGLE: Set to 0 to BYPASS cleaning
        }

        # backward-compat main csv
        self.csv_path = os.path.join(self.index_dir, "index_train.csv")

        # Load CLAP model
        print("Loading CLAP model...")
        
        self.clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID)
        self.clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)

        # pick device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clap_model.to(self.device)
        print("Using device:", self.device)
        print(f"[INFO] Using CLAP backbone: {CLAP_MODEL_ID}")

        # Detect CLAP embedding dimension dynamically (IMPORTANT)
        with torch.no_grad():
            dummy_text = self.clap_processor(
                text=["test"], return_tensors="pt", padding=True
            ).to(self.device)
            dummy_embed = self.clap_model.get_text_features(**dummy_text)
            # Handle both tensor and BaseModelOutputWithPooling returns
            if hasattr(dummy_embed, 'shape'):
                self.embed_dim = dummy_embed.shape[-1]
            elif hasattr(dummy_embed, 'text_embeds') and dummy_embed.text_embeds is not None:
                self.embed_dim = dummy_embed.text_embeds.shape[-1]
            elif hasattr(dummy_embed, 'pooler_output') and dummy_embed.pooler_output is not None:
                self.embed_dim = dummy_embed.pooler_output.shape[-1]
            elif hasattr(dummy_embed, 'last_hidden_state') and dummy_embed.last_hidden_state is not None:
                self.embed_dim = dummy_embed.last_hidden_state.shape[-1]
            elif isinstance(dummy_embed, tuple):
                self.embed_dim = dummy_embed[0].shape[-1]
            else:
                # Fallback to default CLAP dimension
                self.embed_dim = 512
                print(f"[WARN] Could not detect embed_dim, using default: {self.embed_dim}")

        print(f"[INFO] CLAP embedding dim = {self.embed_dim}")

        # Initialize components
        self.df = None
        self.label_encoder = LabelEncoder()
        self.tone_classifier = None
        self.knob_regressor_net = None
        self.knob_scaler = StandardScaler()
        self.tone_embeddings = {}

                
        # 17 parameters
        self.knob_params = [
            "overdrivedrive", "distortiondrive", "distortiontone",
            "eqbass", "eqmid", "eqtreble",
            "chorusrate", "chorusdepth", "chorusmix",
            "delaytime", "delayfeedback", "delaymix",
            "reverbt60", "reverbdamp", "reverbsize", "reverbwet",
            "mastervolume",
        ]        
        
        # Average knob settings per tone type (fallback for inference without dataset)
        self.tone_defaults = {
            'Clean': {
                'distortiondrive': 0.05, 'overdrivedrive': 0.05, 'distortiontone': 0.5,
                'eqbass': 0.45, 'eqmid': 0.5, 'eqtreble': 0.55,
                'chorusrate': 0.3, 'chorusdepth': 0.2, 'chorusmix': 0.15,
                'delaytime': 0.25, 'delayfeedback': 0.2, 'delaymix': 0.1,
                'reverbt60': 0.15, 'reverbdamp': 0.4, 'reverbsize': 0.3, 'reverbwet': 0.2,
                'mastervolume': 0.5
            },
            'Crunch': {
                'distortiondrive': 0.35, 'overdrivedrive': 0.45, 'distortiontone': 0.55,
                'eqbass': 0.5, 'eqmid': 0.6, 'eqtreble': 0.5,
                'chorusrate': 0.25, 'chorusdepth': 0.15, 'chorusmix': 0.1,
                'delaytime': 0.3, 'delayfeedback': 0.25, 'delaymix': 0.15,
                'reverbt60': 0.2, 'reverbdamp': 0.45, 'reverbsize': 0.35, 'reverbwet': 0.25,
                'mastervolume': 0.55
            },
            'High-Gain': {
                'distortiondrive': 0.85, 'overdrivedrive': 0.75, 'distortiontone': 0.6,
                'eqbass': 0.6, 'eqmid': 0.45, 'eqtreble': 0.65,
                'chorusrate': 0.2, 'chorusdepth': 0.1, 'chorusmix': 0.05,
                'delaytime': 0.2, 'delayfeedback': 0.3, 'delaymix': 0.1,
                'reverbt60': 0.15, 'reverbdamp': 0.5, 'reverbsize': 0.25, 'reverbwet': 0.15,
                'mastervolume': 0.6
            },
            'Wellness': {
                'distortiondrive': 0.1, 'overdrivedrive': 0.15, 'distortiontone': 0.45,
                'eqbass': 0.4, 'eqmid': 0.45, 'eqtreble': 0.5,
                'chorusrate': 0.4, 'chorusdepth': 0.35, 'chorusmix': 0.4,
                'delaytime': 0.5, 'delayfeedback': 0.45, 'delaymix': 0.35,
                'reverbt60': 0.65, 'reverbdamp': 0.3, 'reverbsize': 0.7, 'reverbwet': 0.6,
                'mastervolume': 0.5
            }
        }

        self.tone_descriptions = {
            'Clean': [
                "dry clean guitar", 
                "unprocessed electric guitar", 
                "direct input tone", 
                "clean jazz guitar", 
                "crystal clear tone"
            ],
            
            # CRUNCH: Claim "Vintage", "Classic", and "Punk" explicitly
            'Crunch': [
                "classic rock guitar", 
                "vintage british crunch", 
                "70s rock tone", 
                "raw punk rhythm",
                "hard rock overdrive",
                "warm tube breakup"
            ],
            
            # HIGH-GAIN: Claim "Modern", "Heavy", and "Drop-Tune" explicitly
            'High-Gain': [
                "modern metal distortion", 
                "heavy drop tune", 
                "aggressive djent", 
                "modern rock distortion", 
                "scooped heavy metal", 
                "saturated lead tone"
            ],
            
            'Wellness': [
                "massive ambient reverb", 
                "washed out ethereal guitar", 
                "underwater soundscape", 
                "infinite sustain delay", 
                "calm meditation music"
            ]
        }
                                                        
    def _preprocess_dataset(self, df, dataset_name="unknown", thresholds=None, save_dir=None):
        """
        Applies Ghost Parameter cleaning.
        Respects 'clean_data' toggle in self.dataset_toggles.
        """
        # --- BYPASS CHECK ---
        if self.dataset_toggles.get("clean_data", 1) == 0:
            print(f"[PREPROCESSING] Skipped for {dataset_name} (Toggle 'clean_data' is 0)")
            return df

        # Use Central Config if no specific thresholds provided
        if thresholds is None:
            thresholds = self.DEFAULT_CLEANING_THRESHOLDS

        # Build final threshold map
        final_thresholds = {}
        known_families = ["n_chorusmix", "n_delaymix", "n_reverbwet"]
        for key in known_families:
            final_thresholds[key] = thresholds.get(key, 0.05)

        print(f"\n[PREPROCESSING] Cleaning Ghost Parameters for: {dataset_name}")
        print(f"   Thresholds applied: {final_thresholds}")
        
        # Define families: Master -> [Dependents]
        families = [
            ("n_chorusmix", ["n_chorusrate", "n_chorusdepth"]),
            ("n_delaymix", ["n_delaytime", "n_delayfeedback"]),
            ("n_reverbwet", ["n_reverbt60", "n_reverbsize", "n_reverbdamp"])
        ]
        
        total_modified = 0
        df_clean = df.copy()
        
        for master, dependents in families:
            if master not in df_clean.columns:
                continue
            
            thresh = final_thresholds[master]
            mask = df_clean[master] < thresh
            count = mask.sum()
            
            if count > 0:
                print(f"   -> {master}: Cleaning {count} rows where value < {thresh}")
                # 1. Clean Dependents
                for child in dependents:
                    if child in df_clean.columns:
                        df_clean.loc[mask, child] = 0.0
                # 2. Clean Master
                df_clean.loc[mask, master] = 0.0
                total_modified += count
                        
        if total_modified == 0:
            print("   -> No ghost parameters found.")
        else:
            print(f"   -> Done. Total ghost rows cleaned in {dataset_name}: {total_modified}")

            # Save processed copy
            try:
                clean_name = dataset_name[:-4] if dataset_name.endswith(".csv") else dataset_name
                if clean_name.startswith("index_"):
                    save_name = f"{clean_name}_processed.csv"
                else:
                    save_name = f"index_{clean_name}_processed.csv"
                
                target_dir = save_dir if save_dir is not None else self.index_dir
                save_path = os.path.join(target_dir, save_name)
                df_clean.to_csv(save_path, index=False)
                print(f"   -> Saved processed copy to: {save_path}")
            except Exception as e:
                print(f"   [WARN] Could not save processed csv: {e}")
                
        return df_clean

    def _load_and_pad_audio(self, audio_path, target_sr=48000):
        """
        Loads audio (Run 16 Logic).
        - If shorter than self.audio_length: Zero-pads at the end.
        - If longer: Crops to self.audio_length.
        Updates self._stat_padded and self._stat_cropped.
        """
        try:
            y, sr = librosa.load(audio_path, sr=target_sr)
            
            target_samples = int(self.audio_length * target_sr)
            current_samples = len(y)

            if current_samples < target_samples:
                # PAD
                missing = target_samples - current_samples
                y = np.pad(y, (0, missing), mode='constant')
                self._stat_padded += 1
            elif current_samples > target_samples:
                # CROP
                y = y[:target_samples]
                self._stat_cropped += 1
            
            # Count total processed via this method
            self._stat_total_processed += 1
            return y, sr
        except Exception as e:
            print(f"[ERR] Audio loading/padding failed for {audio_path}: {e}")
            return np.zeros(int(self.audio_length * target_sr)), target_sr

    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} samples with {len(self.df['scenario'].unique())} tone types")
        return self.df
            
    def extract_audio_features(self, audio_path, max_samples=None):
        """Extract CLAP embeddings from audio file (WITH PADDING logic)"""
        try:
            # UPDATED: Use new padding loader
            audio, sr = self._load_and_pad_audio(audio_path, target_sr=48000)
            
            # Get CLAP audio embedding
            inputs = self.clap_processor(audios=audio, return_tensors="pt", sampling_rate=48000)
            inputs = {k: v.to(self.device) for k, v in inputs.items()} 

            with torch.no_grad():
                audio_embed = self.clap_model.get_audio_features(**inputs)

            return audio_embed.cpu().numpy().flatten()

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros(self.embed_dim)
            
    def extract_text_features(self, text_list):
        """Extract CLAP embeddings from text descriptions"""
        inputs = self.clap_processor(text=text_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # NEW

        with torch.no_grad():
            text_embed = self.clap_model.get_text_features(**inputs)
            # Handle both tensor and BaseModelOutputWithPooling returns
            if hasattr(text_embed, 'text_embeds'):
                text_embed = text_embed.text_embeds
            elif hasattr(text_embed, 'pooler_output'):
                text_embed = text_embed.pooler_output
            elif hasattr(text_embed, 'last_hidden_state'):
                # Use mean pooling of last hidden state as fallback
                text_embed = text_embed.last_hidden_state.mean(dim=1)
            elif isinstance(text_embed, tuple):
                text_embed = text_embed[0]
            # If it's still not a tensor, try to get the first value
            if not hasattr(text_embed, 'cpu'):
                # Last resort - iterate through the object's values
                for attr in ['text_embeds', 'pooler_output', 'last_hidden_state']:
                    if hasattr(text_embed, attr):
                        val = getattr(text_embed, attr)
                        if val is not None and hasattr(val, 'cpu'):
                            text_embed = val
                            break

        return text_embed.cpu().numpy()
    
    def _discover_datasets(self):
        """
        Find all index_*.csv in self.index_dir.
        Returns dict like:
            {"train": ".../index_train.csv", "temporal": ".../index_temporal.csv", ...}
        """
        datasets = {}
        if not os.path.isdir(self.index_dir):
            print(f"[WARN] index dir not found: {self.index_dir}")
            return datasets

        for fname in os.listdir(self.index_dir):
            if fname.startswith("index_") and fname.endswith(".csv"):
                key = fname[len("index_"):-4]  # drop prefix/suffix
                full_path = os.path.join(self.index_dir, fname)
                datasets[key] = full_path
        return datasets

    def _load_index_csv(self, csv_path):
        """Simple CSV loader to get a DataFrame."""
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERR] could not read {csv_path}: {e}")
            return pd.DataFrame()

    def _dataset_key_to_folder(self, dataset_key: str) -> str:
        """
        Map short keys from index filenames to actual folder names on disk.
        index_train.csv  -> generated_audio/train/...
        Everything else stays the same.
        """
        # This is simplified from CRNN: we only care about 'train'
        if dataset_key == "train":
            return "train"
        return dataset_key


    def _resolve_sample_paths(self, dataset_key, row):
        """
        Build audio/label paths from a row and the dataset name.
        We expect parallel folders:
            generated_audio/<dataset_key>/
            labels_normalised/<dataset_key>/
        (Simplified: No DI logic)
        """
        # audio
        audio_name = row.get("audio_path") or row.get("basename")
        if audio_name and not audio_name.lower().endswith(".wav"):
            audio_name = audio_name + ".wav"
        
        folder_key = self._dataset_key_to_folder(dataset_key)
        audio_path = os.path.join(self.audio_root, folder_key, audio_name) if audio_name else None

        # labels
        label_name = row.get("label_path") or row.get("label_file")
        label_path = None
        if label_name:
            label_path = os.path.join(self.labels_root, folder_key, label_name)

        # No di_path
        return audio_path, label_path
    
            
    def _load_validation_external(self, val_path, max_samples=None):
        """Helper to load validation data from a separate directory structure"""
        print(f"[INFO] Loading external validation data from: {val_path}")
        
        old_index = self.index_dir
        old_audio = self.audio_root
        old_labels = self.labels_root
        old_df = self.df
        
        try:
            self.index_dir = os.path.join(val_path, "indexes")
            self.audio_root = os.path.join(val_path, "generated_audio")
            self.labels_root = os.path.join(val_path, "labels_normalised")
            
            val_csv_path = os.path.join(self.index_dir, "index_validate.csv")
            if not os.path.exists(val_csv_path):
                raise FileNotFoundError(f"Validation index not found at: {val_csv_path}")
            
            self.df = pd.read_csv(val_csv_path)
            
            # [DATA CLEANING] Preprocess Validation Data
            self.df = self._preprocess_dataset(self.df, dataset_name="validate")
            
            print(f"[INFO] Validation index loaded. Samples: {len(self.df)}")

            X_val, y_tone_val, y_knob_val = self.prepare_training_data(
                max_samples_per_class=max_samples,
                dataset_key="validate" 
            )
            return X_val, y_tone_val, y_knob_val
            
        finally:
            self.index_dir = old_index
            self.audio_root = old_audio
            self.labels_root = old_labels
            self.df = old_df
    
    def prepare_training_data(self, max_samples_per_class=None, dataset_key="train"):
        """Prepare training data with audio embeddings and target values"""
        print(f"Preparing training data for dataset '{dataset_key}'...")
        
        # Limit samples if specified
        if max_samples_per_class:
            df_sampled = self.df.groupby('scenario').head(max_samples_per_class)
        else:
            df_sampled = self.df
            
        audio_features = []
        tone_labels = []
        knob_values = []
        
        for idx, row in df_sampled.iterrows():
            print(f"Processing {idx+1}/{len(df_sampled)}: {row['scenario']}")
            
            # Use new resolver to get correct path
            audio_path, label_path = self._resolve_sample_paths(dataset_key, row)
            
            if not audio_path or not os.path.exists(audio_path):
                print(f"[WARN] audio not found: {audio_path}")
                continue
                
            if os.path.exists(audio_path):
                             
                features = self.extract_audio_features(audio_path)
                audio_features.append(features)
                tone_labels.append(row['scenario'])
                
                # Get knob values from normalized columns
                knob_vals = []
                for param in self.knob_params:
                    knob_vals.append(row[f'n_{param}'])
                knob_values.append(knob_vals)
            else:
                print(f"Audio file not found: {audio_path}")
        
        return np.array(audio_features), np.array(tone_labels), np.array(knob_values)
    
    
    def train_models(
        self,
        max_samples_per_class=20,
        epochs=300,
        batch_size=32,
        learning_rate=0.001,
        dataset_toggles: dict = None,
        validate_data_dir=r"C:\Aarush\Project\Clap\data_validate"
    ):
        """Train tone classification and knob regression models using external validation"""
        print("Training models...")

        # Reset Stats
        self._stat_padded = 0
        self._stat_cropped = 0
        self._stat_total_processed = 0

        # start timing
        start_all = time.time()
        self.timing_info = { "start_all": start_all }

        # merge external toggles
        if dataset_toggles is not None:
            self.dataset_toggles.update(dataset_toggles)

        print(f"[INFO] Dataset toggles: {self.dataset_toggles}")

        dataset_map = self._discover_datasets()
        print(f"[INFO] Discovered datasets: {list(dataset_map.keys())}")

        # ---- 1) load MAIN train df ----
        if "train" in dataset_map:
            main_df = self._load_index_csv(dataset_map["train"])
        else:
            main_df = self.load_data() 

        # [DATA CLEANING] Preprocess Main Dataset
        main_df = self._preprocess_dataset(main_df, dataset_name="train")

        all_X = []
        all_y_tones = []
        all_y_knobs = []

        # ---- a) always include train ----
        self.df = main_df 
        X_train_main, y_tones_main, y_knobs_main = self.prepare_training_data(
            max_samples_per_class=max_samples_per_class,
            dataset_key="train"
        )
        all_X.append(X_train_main)
        all_y_tones.append(y_tones_main)
        all_y_knobs.append(y_knobs_main)
        print(f"[INFO] train: {len(X_train_main)} samples prepared.")

        self.timing_info["base_done"] = time.time()

        # ---- b) include all other normal datasets (auto) ----
        for ds_key, ds_path in dataset_map.items():
            if ds_key in ("train", "sens", "inv"):
                continue

            toggle_name = f"{ds_key}_aware"
            if self.dataset_toggles.get(toggle_name, 1) != 1:
                print(f"[INFO] dataset '{ds_key}' present but {toggle_name}=0 → skipping.")
                continue

            print(f"[INFO] Loading extra dataset: '{ds_key}'")
            ds_df = self._load_index_csv(ds_path)
            
            # [DATA CLEANING] Preprocess Additional Dataset
            ds_df = self._preprocess_dataset(ds_df, dataset_name=ds_key)

            self.df = ds_df 
            X_ds, y_ds_tones, y_ds_knobs = self.prepare_training_data(
                max_samples_per_class=max_samples_per_class,
                dataset_key=ds_key
            )
            if len(X_ds) > 0:
                all_X.append(X_ds)
                all_y_tones.append(y_ds_tones)
                all_y_knobs.append(y_ds_knobs)
                print(f"[INFO] {ds_key}: {len(X_ds)} samples prepared and merged into main training pool.")

                if ds_key == "temporal":
                    self.timing_info["temporal_done"] = time.time()
            else:
                print(f"[WARN] dataset '{ds_key}' enabled but produced 0 samples.")

        # restore main df
        self.df = main_df

        # [AUDIO TILING REPORT]
        print(f"\n[AUDIO STATS] Training Phase:")
        print(f"   -> Audio Length Target: {self.audio_length} seconds")
        print(f"   -> Files Zero-Padded:   {self._stat_padded}")
        print(f"   -> Files Cropped:       {self._stat_cropped}")
        print(f"   -> Total Processed:     {self._stat_total_processed}\n")

        # ---- c) merge all supervised rows ----
        X_train = np.concatenate(all_X, axis=0)
        y_tone_train_raw = np.concatenate(all_y_tones, axis=0)
        y_knob_train = np.concatenate(all_y_knobs, axis=0)
        print(f"[INFO] merged supervised datasets → {X_train.shape[0]} total TRAINING samples.")

        if len(X_train) == 0:
            raise ValueError("No training data available")

        # ---- d) Load EXTERNAL VALIDATION data ----
        print(f"[INFO] Loading external validation data...")
        
        # Reset stats for validation reporting
        self._stat_padded = 0
        self._stat_cropped = 0
        
        X_test, y_tone_test_raw, y_knob_test = self._load_validation_external(
            validate_data_dir, 
            max_samples=max_samples_per_class
        )
        
        # [AUDIO TILING REPORT - VALIDATION]
        print(f"\n[AUDIO STATS] Validation Phase:")
        print(f"   -> Files Zero-Padded:   {self._stat_padded}")
        print(f"   -> Files Cropped:       {self._stat_cropped}")
        
        print(f"[INFO] External validation loaded → {X_test.shape[0]} total VALIDATION samples.")

        if len(X_test) == 0:
            raise ValueError("No validation data available. Check path or index_validate.csv")
        
        # ---- e) Encode Labels ----
        # Fit encoder on TRAINING data only, then transform both
        y_tone_train = self.label_encoder.fit_transform(y_tone_train_raw)
        
        # Transform validation (handle unseen labels if necessary, but here we assume consistency)
        try:
            y_tone_test = self.label_encoder.transform(y_tone_test_raw)
        except ValueError as e:
            print("[WARN] Validation set contains labels not seen in training! Falling back to fitting on combined.")
            # Fallback: fit on combined (optional, depending on strictness)
            combined_labels = np.concatenate([y_tone_train_raw, y_tone_test_raw])
            self.label_encoder.fit(combined_labels)
            y_tone_train = self.label_encoder.transform(y_tone_train_raw)
            y_tone_test = self.label_encoder.transform(y_tone_test_raw)

        # keep these so the report writer can use them
        self.X_tone_test = X_test
        self.y_tone_test = y_tone_test

        # Train tone classifier
        from sklearn.ensemble import RandomForestClassifier
        self.tone_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.tone_classifier.fit(X_train, y_tone_train)
        
        # Evaluate tone classifier (console)
        y_pred = self.tone_classifier.predict(X_test)
        print("Tone Classification Report:")
        print(
            classification_report(
                y_tone_test,
                y_pred,
                target_names=self.label_encoder.classes_
            )
        )
        
        # Train neural network for knob parameters
        print("Training neural network for knob parameters...")
        # mark train start time right before NN training
        self.timing_info["train_start"] = time.time()
        self._train_knob_neural_network(
            X_train, X_test, y_knob_train, y_knob_test,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Create text embeddings for each tone type
        print("Creating text embeddings for tone matching...")
        for tone_type, descriptions in self.tone_descriptions.items():
            text_embeds = self.extract_text_features(descriptions)
            self.tone_embeddings[tone_type] = np.mean(text_embeds, axis=0)
                
    def _train_knob_neural_network(
        self,
        X_train,
        X_test,
        y_knob_train,
        y_knob_test,
        epochs=300,
        batch_size=32,
        learning_rate=0.001,
    ):
        """Train neural network for knob parameter prediction and build report."""

        # Normalize input features
        X_train_scaled = self.knob_scaler.fit_transform(X_train)
        X_test_scaled = self.knob_scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_knob_train).to(self.device)
        y_test_tensor = torch.FloatTensor(y_knob_test).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Initialize neural network
        self.knob_regressor_net = KnobParameterNet(
            input_dim=X_train.shape[1],
            hidden_dims=[256, 128, 64],
            output_dim=len(self.knob_params),
            dropout_rate=0.25
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.knob_regressor_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=20, factor=0.5
        )
        
        
        # Training loop
        self.knob_regressor_net.train()
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = -1
        best_regressor_state = None
        
        # History buffers for plotting
        train_loss_history = []
        val_loss_history = []
        
        print(f"Training neural network on {self.device}...")
        
        epochs_run = int(epochs)  # will be updated if early-stopped
        for epoch in range(epochs):
            # Start timer for log matching
            epoch_start = time.time()
            
            total_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.knob_regressor_net(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_loss / max(1, num_batches)
            
            # Validation
            self.knob_regressor_net.eval()
            with torch.no_grad():
                val_predictions = self.knob_regressor_net(X_test_tensor)
                val_loss = criterion(val_predictions, y_test_tensor).item()
            
            scheduler.step(val_loss)
            
            # Save history for plotting
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(val_loss)
            
            # Early stopping & Best Model Capture
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_regressor_state = copy.deepcopy(self.knob_regressor_net.state_dict())
            else:
                patience_counter += 1
                
                if patience_counter >= 30:
                    print(f"Early stopping at epoch {epoch+1}")
                    epochs_run = int(epoch + 1)   # record actual epochs run
                    break
            
            # LOGGING: Matches format "[11:46:42] Epoch X/Y (45.5s): Train Loss = ..."
            if (epoch + 1) % 1 == 0:  # Print every epoch to match log
                current_time = datetime.now().strftime("[%H:%M:%S]")
                epoch_dur = time.time() - epoch_start
                print(
                    f"{current_time} Epoch {epoch+1}/{epochs} ({epoch_dur:.1f}s): "
                    f"Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}"
                )
            
            self.knob_regressor_net.train()

        # ===== load best validation checkpoint before final evaluation =====
        if best_regressor_state is not None:
            self.knob_regressor_net.load_state_dict(best_regressor_state)
            print(f"\n[INFO] Restored best validation weights from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
        else:
            print("\n[WARN] No best-validation checkpoint recorded; using last-epoch weights.")
        
        # Final evaluation
        self.knob_regressor_net.eval()
        with torch.no_grad():
            final_predictions = self.knob_regressor_net(X_test_tensor).cpu().numpy()
        
        # --- PLOTTING SECTION ---
        # === NEW: Save training / validation loss plot ===
        try:
            if len(train_loss_history) > 0 and len(val_loss_history) == len(train_loss_history):
                epochs_axis = np.arange(1, len(train_loss_history) + 1, dtype=int)

                # Best validation epoch (index and 1-based epoch number)
                best_idx = int(np.argmin(val_loss_history))
                best_epoch_num = int(epochs_axis[best_idx])
                best_val_value = float(val_loss_history[best_idx])

                # Final epoch (last point actually run – handles early stopping)
                final_epoch_num = int(epochs_axis[-1])
                final_val_value = float(val_loss_history[-1])

                plt.figure(figsize=(12, 5))
                plt.plot(epochs_axis, train_loss_history, label="Train Loss")
                plt.plot(epochs_axis, val_loss_history, label="Validation Loss")

                # Mark best val point
                plt.scatter(best_epoch_num, best_val_value)
                plt.annotate(
                    f"Best Val: {best_val_value:.5f}\n(Epoch {best_epoch_num})",
                    xy=(best_epoch_num, best_val_value),
                    xytext=(best_epoch_num + 5, best_val_value + 0.0005),
                    arrowprops=dict(arrowstyle="->")
                )

                # Mark final epoch val
                plt.scatter(final_epoch_num, final_val_value)
                plt.annotate(
                    f"Final Val: {final_val_value:.5f}\n(Epoch {final_epoch_num})",
                    xy=(final_epoch_num, final_val_value),
                    xytext=(final_epoch_num + 5, final_val_value + 0.0005),
                    arrowprops=dict(arrowstyle="->")
                )

                plt.xlabel("Epoch")
                plt.ylabel("Loss (MSE)")
                plt.title("Training and Validation Loss")
                plt.legend()
                plt.tight_layout()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                _model = str(REPORT_TUNING.get("model_name", "mlp")).strip().lower() or "mlp"
                
                os.makedirs("models", exist_ok=True)
                loss_plot_name = f"epoch_run_{timestamp}_{_model}.png"
                loss_plot_path = os.path.join("models", loss_plot_name)
                plt.savefig(loss_plot_path, dpi=150)
                plt.close()
                print(f"[INFO] Loss chart saved to: {loss_plot_path}")
        except Exception as e:
            print(f"[WARN] Failed to save loss curve plot: {e}")

        

        # --- UPDATED PRINTING SECTION ---
        print("\nParameter-wise Performance:")
        total_mse = 0.0
        per_param_results = []
        for i, param in enumerate(self.knob_params):
            param_mse = mean_squared_error(y_knob_test[:, i], final_predictions[:, i])
            total_mse += param_mse
            per_param_results.append((param, param_mse))
            # Added (weight=1.00) to match CRNN log style
            print(f"{param}: MSE = {param_mse:.6f} (weight=1.00)")
        
        avg_mse = total_mse / len(self.knob_params)
        
        # Overall R²
        overall_r2 = r2_score(y_knob_test, final_predictions)
        print(f"\nAverage MSE across all parameters: {avg_mse:.6f}")
        print(f"Overall R² Score: {overall_r2:.4f}")

        # mark training done time
        if hasattr(self, "timing_info"):
            self.timing_info["train_done"] = time.time()

        # ===== write report =====
        os.makedirs("models", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _model = str(REPORT_TUNING.get("model_name", "mlp")).strip().lower() or "mlp"
        report_path = os.path.join("models", f"test_evaluation_results_{timestamp}_val_{_model}.txt")
        self.last_report_path = report_path

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("================================================================================\n")
            f.write(f"ToneCraft - VALIDATION SET EVALUATION RESULTS - {str(REPORT_TUNING.get('model_name','mlp')).upper()}\n")
            f.write("================================================================================\n\n")

            # 1) Tone classification performance
            f.write("TONE CLASSIFICATION PERFORMANCE\n")
            f.write("--------------------------------------------------------------------------------\n")
            if (
                hasattr(self, "tone_classifier") and self.tone_classifier is not None
                and hasattr(self, "X_tone_test") and hasattr(self, "y_tone_test")
            ):
                X_tone_test = self.X_tone_test
                y_tone_test = self.y_tone_test
                tone_preds = self.tone_classifier.predict(X_tone_test)
                class_names = list(self.label_encoder.classes_)
                acc = accuracy_score(y_tone_test, tone_preds)
                f.write(f"Overall Accuracy: {acc*100:.2f}%\n\n")

                report_dict = classification_report(
                    y_tone_test,
                    tone_preds,
                    target_names=class_names,
                    output_dict=True,
                )
                f.write("Per-Scenario Metrics:\n")
                f.write(f"{'Scenario Name':<27s}{'Precision':>12s}{'Recall':>10s}{'F1-Score':>12s}{'Support':>10s}\n")
                f.write("-" * 71 + "\n")
                for cname in class_names:
                    metrics_c = report_dict.get(cname, {})
                    f.write(
                        f"{cname:<27s}"
                        f"{metrics_c.get('precision', 0.0):>12.4f}"
                        f"{metrics_c.get('recall', 0.0):>10.4f}"
                        f"{metrics_c.get('f1-score', 0.0):>12.4f}"
                        f"{int(metrics_c.get('support', 0)):>10d}\n"
                    )
                # === Insert Macro/Weighted rows exactly like test_model.py ===
                f.write("-" * 71 + "\n")
                macro = report_dict.get('macro avg', {})
                weighted = report_dict.get('weighted avg', {})

                f.write(
                    f"{'Macro Avg':<27s}"
                    f"{macro.get('precision', 0.0):>12.4f}"
                    f"{macro.get('recall', 0.0):>10.4f}"
                    f"{macro.get('f1-score', 0.0):>12.4f}"
                    f"{int(macro.get('support', 0)):>10d}\n"
                )
                f.write(
                    f"{'Weighted Avg':<27s}"
                    f"{weighted.get('precision', 0.0):>12.4f}"
                    f"{weighted.get('recall', 0.0):>10.4f}"
                    f"{weighted.get('f1-score', 0.0):>12.4f}"
                    f"{int(weighted.get('support', 0)):>10d}\n"
                )
                f.write("\n")

                cm = confusion_matrix(
                    y_tone_test,
                    tone_preds,
                    labels=list(range(len(class_names)))
                )
                f.write("Confusion Matrix (rows = true, cols = pred):\n")
                f.write("                " + "".join([f"{c:>10s}" for c in class_names]) + "\n")
                for i, row_cm in enumerate(cm):
                    f.write(f"{class_names[i]:<15s}" + "".join([f"{v:>10d}" for v in row_cm]) + "\n")
            else:
                f.write("Overall Accuracy: 0.00%\n")
            f.write("\n\n")

            # 2) Parameter prediction performance
            f.write("================================================================================\n")
            f.write("PARAMETER PREDICTION PERFORMANCE\n")
            f.write("================================================================================\n\n")
            f.write("Overall Metrics:\n")
            f.write(f"  Mean Squared Error (MSE):  {avg_mse:.6f}\n")
            overall_mae = mean_absolute_error(
                y_knob_test.reshape(-1),
                final_predictions.reshape(-1)
            )
            f.write(f"  Mean Absolute Error (MAE): {overall_mae:.6f}\n")
            f.write(f"  R² Score:                  {overall_r2:.4f}\n\n")

            f.write(f"{'S.No':>4s}  {'Parameter':<24s}{'MSE':>12s}{'MAE':>12s}{'R²':>10s}\n")
            f.write("-" * 60 + "\n")
            for idx, (name, mse_val) in enumerate(per_param_results, start=1):
                y_true_i = y_knob_test[:, idx - 1]
                y_pred_i = final_predictions[:, idx - 1]
                mae_i = mean_absolute_error(y_true_i, y_pred_i)
                try:
                    r2_i = r2_score(y_true_i, y_pred_i)
                except Exception:
                    r2_i = 0.0
                f.write(
                    f"{idx:>4d}  {name:<24s}{mse_val:>12.6f}{mae_i:>12.6f}{r2_i:>10.4f}\n"
                )

            # ----- group averages (ported from sample) -----
            def _avg_metrics(start_idx, end_idx):
                subset = list(range(start_idx, end_idx + 1))
                mses = []
                maes = []
                r2s = []
                for i in subset:
                    y_true_i = y_knob_test[:, i - 1]
                    y_pred_i = final_predictions[:, i - 1]
                    mses.append(mean_squared_error(y_true_i, y_pred_i))
                    maes.append(mean_absolute_error(y_true_i, y_pred_i))
                    try:
                        r2s.append(r2_score(y_true_i, y_pred_i))
                    except Exception:
                        r2s.append(0.0)
                mse_avg = sum(mses) / len(mses)
                mae_avg = sum(maes) / len(maes)
                r2_avg = sum(r2s) / len(r2s)
                return mse_avg, mae_avg, r2_avg

            f.write("\n")
            # 1–6, 7–16
            mse_1_6, mae_1_6, r2_1_6 = _avg_metrics(1, 6)
            mse_7_16, mae_7_16, r2_7_16 = _avg_metrics(7, 16)
            f.write(f"{'':>4s}  {'Average of 1 to 6':<24s}{mse_1_6:>12.5f}{mae_1_6:>12.5f}{r2_1_6:>10.3f}\n")
            f.write(f"{'':>4s}  {'Average of 7 to 16':<24s}{mse_7_16:>12.5f}{mae_7_16:>12.5f}{r2_7_16:>10.3f}\n")

            # 5 detailed groups
            mse_gain, mae_gain, r2_gain = _avg_metrics(1, 3)
            mse_eq, mae_eq, r2_eq = _avg_metrics(4, 6)
            mse_chorus, mae_chorus, r2_chorus = _avg_metrics(7, 9)
            mse_delay, mae_delay, r2_delay = _avg_metrics(10, 12)
            mse_reverb, mae_reverb, r2_reverb = _avg_metrics(13, 16)

            f.write("\n")
            f.write(f"{'':>4s}  {'Average Gain (1–3)':<24s}{mse_gain:>12.5f}{mae_gain:>12.5f}{r2_gain:>10.3f}\n")
            f.write(f"{'':>4s}  {'Average EQ (4–6)':<24s}{mse_eq:>12.5f}{mae_eq:>12.5f}{r2_eq:>10.3f}\n")
            f.write(f"{'':>4s}  {'Average Chorus (7–9)':<24s}{mse_chorus:>12.5f}{mae_chorus:>12.5f}{r2_chorus:>10.3f}\n")
            f.write(f"{'':>4s}  {'Average Delay (10–12)':<24s}{mse_delay:>12.5f}{mae_delay:>12.5f}{r2_delay:>10.3f}\n")
            f.write(f"{'':>4s}  {'Average Reverb (13–16)':<24s}{mse_reverb:>12.5f}{mae_reverb:>12.5f}{r2_reverb:>10.3f}\n")

            # ----- Performance by Scenario -----
            f.write("\n\n")
            f.write(f"{'':>4s}  {'Performance by Scenario':<24s}{'MSE':>12s}{'MAE':>12s}{'R²':>10s}\n")
            f.write("    " + "-" * 56 + "\n")
            
            # We use the validation set data: y_tone_test, y_knob_test, and final_predictions
            for class_name in self.label_encoder.classes_:
                try:
                    # Get the encoded label for the current class name
                    class_label = self.label_encoder.transform([class_name])[0]
                    
                    # Create a mask to find all rows belonging to this class
                    mask = (y_tone_test == class_label)
                    
                    if np.sum(mask) == 0:
                        # Skip if this class had no samples in the validation set
                        continue
                        
                    # Filter the true and predicted knobs for this class
                    y_true_scenario = y_knob_test[mask]
                    y_pred_scenario = final_predictions[mask]
                    
                    # Calculate metrics for this class
                    scenario_mse = mean_squared_error(y_true_scenario, y_pred_scenario)
                    scenario_mae = mean_absolute_error(y_true_scenario, y_pred_scenario)
                    scenario_r2 = r2_score(y_true_scenario, y_pred_scenario)
                    
                    # Write the formatted row, aligned with the table above
                    f.write(
                        f"{'':>4s}  {class_name:<24s}{scenario_mse:>12.6f}{scenario_mae:>12.6f}{scenario_r2:>10.4f}\n"
                    )
                except Exception as e:
                    # Handle cases where a class might be missing or R2 fails
                    f.write(f"{'':>4s}  {class_name:<24s}{'N/A':>12s}{'N/A':>12s}{'N/A':>10s} (Error: {e})\n")
            
            f.write("\n")
            # -----------------------------------------------------------------------------
            # TUNING FACTORS (auto-filled from actual run; REPORT_TUNING overrides if set)
            # -----------------------------------------------------------------------------
            f.write("================================================================================\n")
            f.write("TUNING FACTORS\n")
            f.write("================================================================================\n")
            try:
                # --- Actual run values available in this scope ---
                _actual_samples = int(X_train.shape[0]) if 'X_train' in locals() else None
                _actual_bs = int(batch_size) if 'batch_size' in locals() else None
                _actual_lr = float(learning_rate) if 'learning_rate' in locals() else None
                # epochs_run was introduced above; falls back to 'epochs' if not present
                _actual_epochs = int(epochs_run) if 'epochs_run' in locals() else (int(epochs) if 'epochs' in locals() else None)

                # Derive dropout from the model (first Dropout layer), else None
                _actual_dropout = None
                try:
                    import torch.nn as _nn
                    for _m in getattr(self.knob_regressor_net, 'network', []):
                        if isinstance(_m, _nn.Dropout):
                            _actual_dropout = float(getattr(_m, 'p', None))
                            break
                except Exception:
                    pass

                # --- User overrides (only if not None) ---
                _rt = REPORT_TUNING
                _val_samples = _rt.get('samples_used_for_training') if _rt.get('samples_used_for_training') is not None else _actual_samples
                _val_bs      = _rt.get('batch_size')             if _rt.get('batch_size')             is not None else _actual_bs
                _val_lr      = _rt.get('learning_rate')          if _rt.get('learning_rate')          is not None else _actual_lr
                _val_epochs  = _rt.get('epochs')                 if _rt.get('epochs')                 is not None else _actual_epochs
                _val_dropout = _rt.get('dropout_rate')           if _rt.get('dropout_rate')           is not None else _actual_dropout

                # 1. Samples used for training (prefer pre-split total if provided)
                # Prefer the pre-split total pushed by the driver; fall back to train set size
                _pre = PRE_SPLIT_TRAIN_SAMPLES
                _fallback = (len(train_dataset) if 'train_dataset' in locals()
                            else (len(train_loader.dataset) if 'train_loader' in locals() and hasattr(train_loader, 'dataset')
                                else None))
                f.write(f"1. Samples used for training: {(_pre if _pre is not None else _fallback)}\n")

                # 2) Datasets used (user-reported) + show current dataset_toggles for convenience
                f.write("2. Datasets used (from user REPORT_TUNING): ")
                if _rt.get('datasets_used') is None:
                    f.write("N/A\n")
                else:
                    f.write(f"{_rt.get('datasets_used')}\n")
                if hasattr(self, 'dataset_toggles'):
                    f.write("   ↳ Current dataset_toggles in code:\n")
                    for k, v in getattr(self, 'dataset_toggles', {}).items():
                        f.write(f"      - {k}: {v}\n")

                # 3–6) batch_size, learning_rate, epochs, dropout_rate
                f.write(f"3. batch-size:     {'' if _val_bs is None else _val_bs}\n")
                f.write(f"4. learning rate:  {'' if _val_lr is None else _val_lr}\n")
                f.write(f"5. epochs:         {'' if _val_epochs is None else _val_epochs}\n")
                f.write(f"6. dropout rate:   {'' if _val_dropout is None else _val_dropout}\n")

                # 7) user_comments
                _uc = _rt.get("user_comments", "")
                f.write(f"7. user comments:  {(_uc if _uc else '—')}\n")

                # 8) Model (also used in filename)
                _mn = str(_rt.get("model_name", "mlp"))
                f.write(f"8. Model:          {_mn}\n")
            except Exception as _e:
                f.write(f"[WARN] Could not render TUNING FACTORS: {_e}\n")

            f.write("\n")
            # TIMING SUMMARY — initial part, rest appended by save_models()
            f.write("================================================================================\n")
            f.write("TIMING SUMMARY\n")
            f.write("================================================================================\n")

            ti = getattr(self, "timing_info", None)
            if ti is not None:
                start_all = ti.get("start_all", None)
                base_done = ti.get("base_done", None)
                train_start = ti.get("train_start", None)
                train_done = ti.get("train_done", None)

                def _fmt_hm(seconds):
                    if seconds is None:
                        return "N/A"
                    mins = int(round(seconds / 60.0))
                    hrs = mins // 60
                    mm = mins % 60
                    return f"{hrs:02d}:{mm:02d} (h:mm)"

                if start_all and base_done:
                    f.write(f"1. Base train dataset processed: {_fmt_hm(base_done - start_all)}\n")
                else:
                    f.write("1. Base train dataset processed: N/A\n")

                f.write("2. Temporal dataset processed:   N/A (not used in this file)\n")
                f.write("3. Sensitivity / invariance:     N/A (not used in this file)\n")

                if train_start and train_done:
                    f.write(f"4. Model training (NN):          {_fmt_hm(train_done - train_start)}\n")
                else:
                    f.write("4. Model training (NN):          N/A\n")
            else:
                f.write("Timing info not available.\n")

        print(f"\n📄 Evaluation report saved to: {report_path}")
    
    def classify_audio(self, audio_path):
        """Classify audio file and predict knob settings"""
        # Extract audio features
        audio_features = self.extract_audio_features(audio_path)

        # --- NEW: PRINT INFER INFO ONCE (Matches CRNN Log) ---
        if not hasattr(self, "_did_print_infer_info"):
            self._did_print_infer_info = True
            # MLP uses 512 dim (CLAP only), so we report 512
            print(f"[INF] infer base_feat dim = {self.embed_dim}, trained feature_dim = {self.embed_dim}")
        # -----------------------------------------------------

        audio_features = audio_features.reshape(1, -1)
        
        # Predict tone type
        tone_pred = self.tone_classifier.predict(audio_features)[0]
        tone_name = self.label_encoder.inverse_transform([tone_pred])[0]
        tone_prob = self.tone_classifier.predict_proba(audio_features)[0]
        confidence = np.max(tone_prob)
        
        # Predict knob settings using neural network
        knob_settings = {}
        if self.knob_regressor_net is not None:
            # Normalize input features
            audio_features_scaled = self.knob_scaler.transform(audio_features)
            audio_tensor = torch.FloatTensor(audio_features_scaled).to(self.device)
            
            self.knob_regressor_net.eval()
            with torch.no_grad():
                knob_predictions = self.knob_regressor_net(audio_tensor).cpu().numpy()[0]
            
            for i, param in enumerate(self.knob_params):
                knob_settings[param] = float(knob_predictions[i])  # Already constrained by sigmoid
        
        else:
            # Fallback to defaults (must not require dataset at inference)
            knob_settings = self.tone_defaults.get(tone_name, self.tone_defaults['Clean'])
        
        return {
            'tone_type': tone_name,
            'confidence': confidence,
            'knob_settings': knob_settings
        }
    
    def match_text_to_tone(self, text_description):
        """
        Match text description directly to knob parameters using neural network.
        This allows for infinite variations based on text input, not just 4 predefined tones.
        """
        # Extract text embedding from CLAP
        text_embed = self.extract_text_features([text_description])[0]
        text_embed = text_embed.reshape(1, -1)
        
        # Predict knob settings directly using neural network
        knob_settings = {}
        if self.knob_regressor_net is not None:
            # Normalize input features (same as audio)
            text_embed_scaled = self.knob_scaler.transform(text_embed)
            text_tensor = torch.FloatTensor(text_embed_scaled).to(self.device)
            
            self.knob_regressor_net.eval()
            with torch.no_grad():
                knob_predictions = self.knob_regressor_net(text_tensor).cpu().numpy()[0]
            
            for i, param in enumerate(self.knob_params):
                knob_settings[param] = float(knob_predictions[i])
        else:
            # Fallback: use tone classification if neural network not available
            return self._match_text_to_tone_fallback(text_description)
        
        # Also calculate tone type for reference (optional)
        similarities = {}
        for tone_type, tone_embed in self.tone_embeddings.items():
            similarity = np.dot(text_embed.flatten(), tone_embed) / (
                np.linalg.norm(text_embed) * np.linalg.norm(tone_embed)
            )
            similarities[tone_type] = similarity
        
        # Handle empty tone_embeddings case
        if similarities:
            best_tone = max(similarities, key=similarities.get)
            confidence = float(similarities[best_tone])
        else:
            best_tone = "Custom"
            confidence = 0.5
        
        return {
            'tone_type': best_tone,  # For reference only
            'confidence': confidence,
            'knob_settings': knob_settings,
            'all_similarities': similarities
        }
    
    def _match_text_to_tone_fallback(self, text_description):
        """Fallback method using tone classification (old approach)"""
        text_embed = self.extract_text_features([text_description])[0]
        
        similarities = {}
        for tone_type, tone_embed in self.tone_embeddings.items():
            similarity = np.dot(text_embed, tone_embed) / (
                np.linalg.norm(text_embed) * np.linalg.norm(tone_embed)
            )
            similarities[tone_type] = similarity
        
        best_tone = max(similarities, key=similarities.get)
        confidence = similarities[best_tone]
        
        
        # Fallback: use predefined tone defaults (must not require dataset at inference)
        knob_settings = self.tone_defaults.get(best_tone, self.tone_defaults['Clean'])
        

        return {
            'tone_type': best_tone,
            'confidence': confidence,
            'knob_settings': knob_settings,
            'all_similarities': similarities
        }

    def generate_detailed_csv_reports(self, save_dir="models"):
        """
        Generates prediction and error CSVs for all processed datasets.
        """
        print("\n[INFO] Generating detailed prediction and error CSV reports...")
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Identify datasets to process
        dataset_map = self._discover_datasets()
        datasets_to_process = []
        
        # Always process 'train' if present
        if "train" in dataset_map:
            datasets_to_process.append(("train", dataset_map["train"]))
            
        for ds_key, ds_path in dataset_map.items():
            if ds_key == "train": continue
            
            # Check if this dataset was enabled via toggles
            toggle_name = f"{ds_key}_aware"
            if self.dataset_toggles.get(toggle_name, 0) == 1:
                datasets_to_process.append((ds_key, ds_path))
        
        # Ensure model is in eval mode
        self.knob_regressor_net.eval()
        
        for key, csv_path in datasets_to_process:
            print(f"  -> Processing {key} ({os.path.basename(csv_path)})...")
            try:
                df = pd.read_csv(csv_path)

                # --- NEW: CALL PREPROCESSING TO TRIGGER LOGS ---
                # This prints "[PREPROCESSING] Cleaning Ghost Parameters..."
                df = self._preprocess_dataset(df, dataset_name=key)
                # -----------------------------------------------
                
                # Collect features for all rows
                features_list = []
                
                for idx, row in df.iterrows():
                    audio_path, _ = self._resolve_sample_paths(key, row)
                    
                    if audio_path and os.path.exists(audio_path):
                        # Extract CLAP features (512,)
                        feat = self.extract_audio_features(audio_path)
                        features_list.append(feat)
                    else:
                        features_list.append(np.zeros(self.embed_dim))
                
                if not features_list:
                    print(f"  [WARN] No rows processed for {key}.")
                    continue
                    
                # Prepare batch for prediction
                X_np = np.array(features_list)
                X_scaled = self.knob_scaler.transform(X_np)
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                
                # Predict
                with torch.no_grad():
                    preds_tensor = self.knob_regressor_net(X_tensor)
                    preds_np = preds_tensor.cpu().numpy()
                
                # Prepare Output DataFrames
                df_pred = df.copy()
                df_error = df.copy()
                
                # Columns to update
                knob_cols = [f"n_{p}" for p in self.knob_params]
                
                # 1. Update Prediction CSV
                df_pred[knob_cols] = preds_np
                
                # 2. Update Error CSV (MAE)
                actuals = df[knob_cols].apply(pd.to_numeric, errors='coerce').fillna(0.5).values
                errors = np.abs(preds_np - actuals)
                df_error[knob_cols] = errors
                
                # Save files
                base_name = os.path.splitext(os.path.basename(csv_path))[0]
                
                path_pred = os.path.join(save_dir, f"{base_name}_prediction_mlp.csv")
                path_err = os.path.join(save_dir, f"{base_name}_error_mlp.csv")
                
                df_pred.to_csv(path_pred, index=False, float_format='%.5f')
                df_error.to_csv(path_err, index=False, float_format='%.5f')
                
                print(f"     Saved: {os.path.basename(path_pred)}")
                print(f"     Saved: {os.path.basename(path_err)}")
                
            except Exception as e:
                print(f"  [ERR] Failed to generate reports for {key}: {e}")

    def save_models(self, save_dir="models"):
        """Save trained models and update timing info (to match sample)."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        with open(os.path.join(save_dir, 'tone_classifier.pkl'), 'wb') as f:
            pickle.dump(self.tone_classifier, f)
        
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save neural network state dict
        if self.knob_regressor_net is not None:
            torch.save(
                self.knob_regressor_net.state_dict(),
                os.path.join(save_dir, 'knob_regressor_net.pth')
            )
        
        # Save scaler
        with open(os.path.join(save_dir, 'knob_scaler.pkl'), 'wb') as f:
            pickle.dump(self.knob_scaler, f)
        
        with open(os.path.join(save_dir, 'tone_embeddings.pkl'), 'wb') as f:
            pickle.dump(self.tone_embeddings, f)
        
        # --- CHANGE START ---
        # Generate the requested CSV reports for all processed datasets
        self.generate_detailed_csv_reports(save_dir=save_dir)
        # --- CHANGE END ---

        save_end = time.time()
        print(f"Models saved to {save_dir}/")

        # record timing like sample does
        if not hasattr(self, "timing_info"):
            self.timing_info = {}
        self.timing_info["models_saved"] = save_end
        if "start_all" in self.timing_info:
            self.timing_info["all_done"] = save_end

        # append final timing rows to the last report if we have it
        if hasattr(self, "last_report_path") and os.path.exists(self.last_report_path):
            try:
                def _fmt_hm(seconds):
                    if seconds is None:
                        return "N/A"
                    mins = int(round(seconds / 60.0))
                    hrs = mins // 60
                    mm = mins % 60
                    return f"{hrs:02d}:{mm:02d} (h:mm)"

                with open(self.last_report_path, "a", encoding="utf-8") as f:
                    start_all = self.timing_info.get("start_all", None)
                    models_saved = self.timing_info.get("models_saved", None)
                    all_done = self.timing_info.get("all_done", None)

                    f.write(
                        "5. Models saved:                 "
                        + (_fmt_hm(models_saved - start_all) if (models_saved and start_all) else "N/A")
                        + "\n"
                    )
                    if all_done and start_all:
                        f.write(
                            "6. Total time for all steps:     "
                            + _fmt_hm(all_done - start_all)
                            + "\n"
                        )
                    else:
                        f.write("6. Total time for all steps:     N/A\n")
            except Exception as e:
                print(f"[WARN] could not append timing info to report: {e}")
    
    def load_models(self, save_dir="models"):
        """Load pre-trained models"""
        with open(os.path.join(save_dir, 'tone_classifier.pkl'), 'rb') as f:
            self.tone_classifier = pickle.load(f)
        
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load neural network
        net_path = os.path.join(save_dir, 'knob_regressor_net.pth')
        if os.path.exists(net_path):
            print(f"[LOAD] Loading knob_regressor_net from {net_path}", flush=True)
            self.knob_regressor_net = KnobParameterNet(
                input_dim=self.embed_dim, 
                hidden_dims=[256, 128, 64], 
                output_dim=len(self.knob_params)
            ).to(self.device)
            self.knob_regressor_net.load_state_dict(torch.load(net_path, map_location=self.device))
            self.knob_regressor_net.eval()
            print(f"[LOAD] knob_regressor_net loaded successfully", flush=True)
        else:
            print(f"[WARN] knob_regressor_net.pth not found at {net_path}", flush=True)
        
        # Load scaler
        with open(os.path.join(save_dir, 'knob_scaler.pkl'), 'rb') as f:
            self.knob_scaler = pickle.load(f)
        print(f"[LOAD] knob_scaler loaded", flush=True)
        
        tone_embed_path = os.path.join(save_dir, 'tone_embeddings.pkl')
        if os.path.exists(tone_embed_path):
            with open(tone_embed_path, 'rb') as f:
                self.tone_embeddings = pickle.load(f)
            print(f"[LOAD] tone_embeddings loaded with {len(self.tone_embeddings)} tones: {list(self.tone_embeddings.keys())}", flush=True)
        else:
            print(f"[WARN] tone_embeddings.pkl not found at {tone_embed_path}", flush=True)
            self.tone_embeddings = {}
        
        # NOTE: Do NOT load dataset here. Inference must work without index CSVs.
        # Training code will call load_data() explicitly.
        pass
        
        print(f"Models loaded from {save_dir}/", flush=True)

def main():
    # Initialize classifier
    classifier = MusicToneClassifier()
    
    # Load data
    classifier.load_data()
    
    # Train models (using subset for faster training)
    classifier.train_models(max_samples_per_class=15)
    
    
    # Save models
    classifier.save_models()
    
    print("\nTraining completed! Models saved.")
    
    # Test with a sample audio file
    if len(classifier.df) > 0:
        sample_audio = os.path.join(classifier.audio_dir, classifier.df.iloc[0]['audio_path'])
        if os.path.exists(sample_audio):
            print(f"\nTesting with sample audio: {sample_audio}")
            result = classifier.classify_audio(sample_audio)
            print(f"Predicted tone: {result['tone_type']} (confidence: {result['confidence']:.3f})")
            print("Knob settings:")
            for param, value in result['knob_settings'].items():
                print(f"  {param}: {value:.3f}")
    
    # Test text matching
    print(f"\nTesting text matching with 'heavy metal':")
    text_result = classifier.match_text_to_tone("heavy metal")
    print(f"Matched tone: {text_result['tone_type']} (confidence: {text_result['confidence']:.3f})")
    print("Recommended knob settings:")
    for param, value in text_result['knob_settings'].items():
        print(f"  {param}: {value:.3f}")

if __name__ == "__main__":
    main()
