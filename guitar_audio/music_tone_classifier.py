#!/usr/bin/env python3
"""
Music Tone Classifier using LAION CLAP model Aarush_model_v3_base dataset
This system can:
1. Classify audio files to tone effects and provide knob settings
2. Match text descriptions to appropriate tone effects and knob settings
Model: crnn+biLSTM
Model settings comments: This is a CRNN model
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import warnings
import copy
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import random
warnings.filterwarnings('ignore')

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

# Additional deterministic guard (safe to ignore if unsupported)
#try:
#    torch.use_deterministic_algorithms(True)
#except Exception:
#    pass

# -----------------------------------------------------------------------------
# Imports (Existing...)
# -----------------------------------------------------------------------------


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
    "user_comments": "No audio tiling, zero padding audio length 5s, New base dataset, threshold and cleaning logic changed, dropoutrate 0.25",
    "model_name": "music_tone_classifier_crnn_final_base_run44",        # controls filename suffix _val_<model>
}

# -----------------------------------------------------------------------------
# FEATURE / CLASSIFIER USER SETTINGS (User-editable; affects training & inference)
# -----------------------------------------------------------------------------
# 1) Feature usage flexibility (0/1) for REGRESSION and CLASSIFICATION separately.
#    - clap: CLAP audio embedding (vector)
#    - mel : log-mel -> MelCRNNEncoder -> mel_embed (vector)
FEATURE_USAGE = {
    # Regression features
    "reg_use_clap": 0,
    "reg_use_mel":  1,

    # Classification features (RF or NN-head)
    "cls_use_clap": 0,
    "cls_use_mel":  1,
}

# 2) Classifier choice:
#    "rf"      -> current RandomForestClassifier workflow (aligned after E2E)
#    "nn_head" -> a lightweight vector classifier head trained on the same fused vector
CLASSIFIER_SETTINGS = {
    "classifier_type": "nn_head",  # "rf" or "nn_head"
}


class MelCRNNEncoder(nn.Module):
    """
    Phase 2 (compressed): 2D-CNN → per-timestep projection → BiLSTM → pooled embedding.

    Problem we saw: 2D-CNN was producing (B, 128, 24, T), which is 128*24=3072 dims per timestep.
    That was too big for our current data/optimizer and slightly hurt accuracy.

    Fix: project 3072 → 512 per timestep, THEN run the BiLSTM on 512.
    """
    def __init__(self, n_mels=96, out_dim=128, rnn_hidden=128, rnn_layers=1):
        super(MelCRNNEncoder, self).__init__()
        self.n_mels = n_mels

        # 1) CNN over mel
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),   # 96 -> 48

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),   # 48 -> 24

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # after CNN: (B, 128, 24, T_out)
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers

        # 2) compress per-timestep feature 3072 -> 512
        self.time_proj_in = 128 * (self.n_mels // 4)   # 128 * 24 = 3072
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_proj_in, 512),
            nn.ReLU(),
            nn.Dropout(0.25)  # <--- NEW: High dropout here is critical
        )

        # 3) BiLSTM now runs on 512-d tokens instead of 3072-d
        self.bi_lstm = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.25 if rnn_layers > 1 else 0.0 # Standard LSTM dropout
        )

        # 4) final projection to mel_out_dim (128)
        self.out_proj = nn.Linear(rnn_hidden * 2, out_dim)

    def forward(self, x):
        """
        x: (B, 1, n_mels, T)
        """
        B = x.size(0)
        x = self.cnn(x)                       # (B, 128, 24, T_out)
        B, C, H, W = x.size()                 # H=24
        # (B, 128, 24, T_out) -> (B, T_out, 128*24)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T_out, C, H)
        x = x.view(B, W, C * H)                 # (B, T_out, 3072)

        # project each timestep to 512 (Sequential now handles ReLU+Dropout)
        x = self.time_proj(x)                   # (B, T_out, 512)

        # BiLSTM over time
        lstm_out, _ = self.bi_lstm(x)           # (B, T_out, 2*rnn_hidden)

        # temporal pooling
        lstm_pooled = lstm_out.mean(dim=1)      # (B, 2*rnn_hidden)

        # final mel_out_dim-D vector (e.g. 256)
        out = self.out_proj(lstm_pooled)        # (B, out_dim)
        return out


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
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer with sigmoid activation to constrain outputs to [0, 1]
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ToneClassifierHead(nn.Module):
    """
    Lightweight vector classifier head for tone classification.
    This is NOT a new audio NN; it consumes the same fused vector that RF uses.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

class EndToEndDataset(torch.utils.data.Dataset):
    """
    Custom Dataset to load data. 
    Updated to support RAM Caching (serving from a list) or Disk (serving from paths).
    """ 
    def __init__(self, classifier_instance, data_source, y_knob=None, mode="ram"):
        """
        Args:
            classifier_instance: reference to parent class
            data_source: If mode="ram", this is a list of tuples (mel, static, y).
                         If mode="disk", this is a list of file paths.
            y_knob: Labels (only used if mode="disk")
            mode: "ram" or "disk"
        """
        self.classifier = classifier_instance
        self.mode = mode
        self.data_source = data_source
        self.y_knob = y_knob
        
    def __len__(self):
        return len(self.data_source)
        
    def __getitem__(self, idx):
        # --- RAM MODE (OPTIMIZED) ---
        if self.mode == "ram":
            # data_source[idx] is already (mel_tensor, static_tensor, label_tensor)
            return self.data_source[idx]

        # --- DISK MODE (LEGACY/FALLBACK) ---
        audio_path = self.data_source[idx]
        
        try:
            # 1. Infer the dataset key from the audio path
            audio_name = os.path.basename(audio_path)
            
            # IMPROVED PATH LOGIC: Search for 'generated_audio' root folder
            path_parts = audio_path.split(os.sep)
            
            # Find the dataset key
            try:
                audio_root_index = path_parts.index(os.path.basename(self.classifier.audio_root))
                dataset_key = path_parts[audio_root_index + 1]
            except ValueError:
                # print(f"[WARN] Could not parse dataset key from path: {audio_path}")
                dataset_key = "train" 

            # 2. Get the cache file path using the robust key
            cache_path, _ = self.classifier._get_cache_path(dataset_key, audio_name)
            
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Cache file missing: {cache_path}")
            
            # 3. Load the cached features
            data = np.load(cache_path)

            # Respect regression feature toggles in disk mode
            use_mel  = (self.classifier.feature_usage.get("reg_use_mel", 1) == 1)
            use_clap = (self.classifier.feature_usage.get("reg_use_clap", 1) == 1)

            # --- Mel (only if enabled) ---
            if use_mel:
                if "mel_feat" not in data:
                    raise KeyError(f"Cache missing mel_feat but reg_use_mel=1: {cache_path}")
                mel_spec = data["mel_feat"]

                # --- SHAPE CHECK: MUST BE 2D SPECTROGRAM ---
                if mel_spec.ndim != 2:
                    raise ValueError(
                        f"Invalid cached mel_feat shape {mel_spec.shape} in {cache_path}; "
                        f"expected 2D (n_mels, T)."
                    )

                mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)  # (1, n_mels, T)
            else:
                # Create a safe zero mel tensor (won't be used downstream when reg_use_mel=0)
                target_width = self.classifier.mel_time_steps
                mel_tensor = torch.zeros(1, self.classifier.mel_n_mels, target_width)

            # --- CLAP/static (only if enabled) ---
            if use_clap:
                if "static_feat" not in data:
                    raise KeyError(f"Cache missing static_feat but reg_use_clap=1: {cache_path}")
                static_tensor = torch.FloatTensor(data["static_feat"])
            else:
                # Create a safe zero static tensor (won't be used downstream when reg_use_clap=0)
                clap_dim = getattr(self.classifier, "clap_dim", 512)
                static_tensor = torch.zeros(clap_dim)

            # Label
            label_tensor = torch.tensor(self.y_knob[idx], dtype=torch.float32)

            return mel_tensor, static_tensor, label_tensor
        
        except Exception as e:
            # Fallback to zero tensors if loading fails
            print(f"[ERR] Load failure for {audio_path}: {e}")
            
            # --- UPDATED FALLBACK TO MATCH DYNAMIC WIDTH ---
            target_width = self.classifier.mel_time_steps
            mel_tensor = torch.zeros(1, self.classifier.mel_n_mels, target_width)

            # Static tensor should always represent CLAP/static dimension
            clap_dim = getattr(self.classifier, "clap_dim", 512)
            static_tensor = torch.zeros(clap_dim)

            knob_tensor = torch.zeros(len(self.classifier.knob_params))

            return mel_tensor, static_tensor, knob_tensor
                
class MusicToneClassifier:
    
    # --- CENTRAL CONFIGURATION START ---
    # Change thresholds here, and it applies to Train, Validate, Test, and Reports.
    DEFAULT_CLEANING_THRESHOLDS = {
        'n_chorusmix': 0.10,   # UPDATED: Was 0.05
        'n_delaymix': 0.20,    # UPDATED: Was 0.05
        'n_reverbwet': 0.25,   # UPDATED: Was 0.20
    }
    # --- CENTRAL CONFIGURATION END ---

    def __init__(self, data_dir="Data", audio_length=5.0): # <--- Added audio_length arg
        # === base folders (expandable) ===
        self.data_dir = data_dir
        self.index_dir = os.path.join(data_dir, "indexes")                 # index_*.csv
        self.audio_root = os.path.join(data_dir, "generated_audio")        # generated_audio/train, /temporal...
        self.labels_root = os.path.join(data_dir, "labels_normalised")     # labels_normalised/train, ...
        

        # NEW: Folder to save pre-extracted Mel-spectrogram and CLAP features
        self.cache_root = os.path.join(data_dir, "cached_features_crnn") # <--- NEW

        # --- AUDIO CONFIGURATION (Option A) ---
        self.audio_length = audio_length
        self.target_sr = 48000
        self.mel_hop_length = 256
        
        # Calculate expected Log-Mel width dynamically: 
        # Formula: 1 + (SR * Duration / Hop)
        # E.g. 5.0s -> 938 frames; 10.0s -> 1876 frames
        self.mel_time_steps = 1 + int(self.audio_length * self.target_sr / self.mel_hop_length)
        
        # Statistics Counters (Internal)
        self._stat_padded = 0
        self._stat_cropped = 0
        self._stat_total_processed = 0

        # Phase 3: dataset toggle matrix (these are now your defaults)
        self.dataset_toggles = {
            "temporal_aware": 0,
            "tempboost_aware": 0,
            "static_aware": 0,
            "clean_data": 1,  # <--- NEW TOGGLE: Set to 0 to BYPASS cleaning
        }

        # --- Feature usage & classifier choice (from user tables above) ---
        self.feature_usage = copy.deepcopy(FEATURE_USAGE)
        self.classifier_type = str(CLASSIFIER_SETTINGS.get("classifier_type", "rf")).strip().lower()

        # Base dims (CLAP always 512 in your setup)
        self.clap_dim = 512

        # NN-head classifier (optional)
        self.tone_classifier_nn = None
        
        
        # backward-compat main csv
        self.csv_path = os.path.join(self.index_dir, "index_train.csv")

        # === CLAP ===
        print("Loading CLAP model...")
        self.clap_model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
        self.clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clap_model.to(self.device)
        print("Using device:", self.device)

        # === Phase 2 CRNN for log-mel ===
        self.mel_n_mels = 96
        self.mel_out_dim = 256
        self.mel_encoder = MelCRNNEncoder(
            n_mels=self.mel_n_mels,
            out_dim=self.mel_out_dim
        ).to(self.device)

        # Feature dims depend on user toggles
        self.feature_dim = self._get_feature_dim_for_task("reg")   # regression dim
        self.cls_feature_dim = self._get_feature_dim_for_task("cls")  # classification dim

        # will be set later
        self.df = None
        self.label_encoder = LabelEncoder()
        self.tone_classifier = None
        self.knob_regressor_net = None
        self.knob_scaler = StandardScaler()
        self.tone_embeddings = {}

        # --- FIX: Hardcode dimensions so load_models works without training first ---
        # CLAP (512) + Bi-LSTM (128) = 640
        #self.feature_dim = 640

        # 17 parameters in your order
        self.knob_params = [
            "overdrivedrive",
            "distortiondrive",
            "distortiontone",
            "eqbass",
            "eqmid",
            "eqtreble",
            "chorusrate",
            "chorusdepth",
            "chorusmix",
            "delaytime",
            "delayfeedback",
            "delaymix",
            "reverbt60",
            "reverbdamp",
            "reverbsize",
            "reverbwet",
            "mastervolume",
        ]

        # Average knob settings per tone type (fallback for inference without dataset)
        self.tone_defaults = {
            'Clean': {
                'distortiondrive': 0.05, 'overdrivedrive': 0.05, 'distortiontone': 0.5,
                'eqbass': 0.45, 'eqmid': 0.5, 'eqtreble': 0.55,
                'chorusrate': 0.3, 'chorusdepth': 0.2, 'chorusmix': 0.15,
                'delaytime': 0.25, 'delayfeedback': 0.2, 'delaymix': 0.1,
                'reverbt60': 0.15, 'reverbsize': 0.3, 'reverbwet': 0.2, 'reverbdamp': 0.4,
                'mastervolume': 0.5
            },
            'Crunch': {
                'distortiondrive': 0.35, 'overdrivedrive': 0.45, 'distortiontone': 0.55,
                'eqbass': 0.5, 'eqmid': 0.6, 'eqtreble': 0.5,
                'chorusrate': 0.25, 'chorusdepth': 0.15, 'chorusmix': 0.1,
                'delaytime': 0.3, 'delayfeedback': 0.25, 'delaymix': 0.15,
                'reverbt60': 0.2, 'reverbsize': 0.35, 'reverbwet': 0.25, 'reverbdamp': 0.45,
                'mastervolume': 0.55
            },
            'High-Gain': {
                'distortiondrive': 0.85, 'overdrivedrive': 0.75, 'distortiontone': 0.6,
                'eqbass': 0.6, 'eqmid': 0.45, 'eqtreble': 0.65,
                'chorusrate': 0.2, 'chorusdepth': 0.1, 'chorusmix': 0.05,
                'delaytime': 0.2, 'delayfeedback': 0.3, 'delaymix': 0.1,
                'reverbt60': 0.15, 'reverbsize': 0.25, 'reverbwet': 0.15, 'reverbdamp': 0.5,
                'mastervolume': 0.6
            },
            'Wellness': {
                'distortiondrive': 0.1, 'overdrivedrive': 0.15, 'distortiontone': 0.45,
                'eqbass': 0.4, 'eqmid': 0.45, 'eqtreble': 0.5,
                'chorusrate': 0.4, 'chorusdepth': 0.35, 'chorusmix': 0.4,
                'delaytime': 0.5, 'delayfeedback': 0.45, 'delaymix': 0.35,
                'reverbt60': 0.65, 'reverbsize': 0.7, 'reverbwet': 0.6, 'reverbdamp': 0.3,
                'mastervolume': 0.5
            }
        }

        # text descriptions
        self.tone_descriptions = {
            "Clean": [
                "clean guitar tone", "pristine sound", "clear guitar", "clean amp",
                "transparent tone", "studio clean", "jazz clean tone", "clean electric guitar"
            ],
            "Crunch": [
                "crunchy guitar", "crunch tone", "slight overdrive", "edge of breakup",
                "vintage crunch", "tube crunch", "bluesy crunch", "classic rock crunch"
            ],
            "High-Gain": [
                "heavy metal", "high gain distortion", "metal guitar", "aggressive tone",
                "brutal distortion", "heavy distortion", "modern metal", "saturated distortion"
            ],
            "Wellness": [
                "ambient guitar", "spacey tone", "ethereal guitar", "reverb heavy",
                "atmospheric guitar", "dreamy tone", "shoegaze guitar", "ambient soundscape"
            ]
        }


    def _discover_datasets(self):
        """
        Find all index_*.csv in self.index_dir.
        Returns dict like:
            {"train": ".../index_train.csv", "sens": ".../index_sens.csv", "inv": ".../index_inv.csv", ...}
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
        index_sens.csv   -> generated_audio/sensitivity/...
        index_inv.csv    -> generated_audio/invariance/...
        index_train.csv  -> generated_audio/train/...
        Everything else stays the same.
        """
        return dataset_key


    def _resolve_sample_paths(self, dataset_key, row):
        """
        Build audio/label/di paths from a row and the dataset name.
        We expect parallel folders:
            generated_audio/<dataset_key>/
            labels_normalised/<dataset_key>/
            
        """
        # audio
        audio_name = row.get("audio_path") or row.get("basename")
        if audio_name and not audio_name.lower().endswith(".wav"):
            audio_name = audio_name + ".wav"
        
        folder_key = self._dataset_key_to_folder(dataset_key)
        audio_path = os.path.join(self.audio_root, folder_key, audio_name) if audio_name else None

        # labels: many of your indexes already have all 17 n_* columns,
        # so label_path may not be used, but we keep it for future sets.
        label_name = row.get("label_path") or row.get("label_file")
        label_path = None
        if label_name:
            if not label_name.lower().endswith(".json"):
                # if you use .csv you can change this
                pass
            label_path = os.path.join(self.labels_root, folder_key, label_name)

        
        return audio_path, label_path

    def _get_cache_path(self, dataset_key, audio_name):
        """Generates a cache path for a specific sample."""
        # Use a path structure that includes the dataset key
        ds_cache_dir = os.path.join(self.cache_root, self._dataset_key_to_folder(dataset_key))
        # Use a consistent filename based on the audio name
        cache_name = audio_name.replace('.wav', '.npz')
        return os.path.join(ds_cache_dir, cache_name), ds_cache_dir

    
    def _load_and_pad_audio(self, audio_path, target_sr=48000):
        """
        Loads audio. 
        - If shorter than self.audio_length: Zero-pads at the end (NO Tiling).
        - If longer: Crops to self.audio_length.
        Updates self._stat_padded and self._stat_cropped.
        """
        try:
            # Load full duration (sr is managed by class config usually, but we accept arg)
            y, sr = librosa.load(audio_path, sr=target_sr)
            
            target_samples = int(self.audio_length * target_sr)
            current_samples = len(y)

            if current_samples < target_samples:
                # PAD (Add zeros to the end)
                missing = target_samples - current_samples
                y = np.pad(y, (0, missing), mode='constant')
                
                # Update stat
                self._stat_padded += 1
            elif current_samples > target_samples:
                # CROP (Cut to target length)
                y = y[:target_samples]
                
                # Update stat
                self._stat_cropped += 1
                
            return y, sr
        except Exception as e:
            print(f"[ERR] Audio loading/padding failed for {audio_path}: {e}")
            # Return silent array of correct length to prevent crash
            return np.zeros(int(self.audio_length * target_sr)), target_sr
            
    def _extract_logmel_spectrogram(self, audio_path):
        """
        Load audio from disk (WITH PADDING) and return a float32 log-mel spectrogram.
        """
        try:
            # UPDATED: Use new padding loader
            audio, sr = self._load_and_pad_audio(audio_path, target_sr=48000)
            
            mel = self._extract_logmel(audio, sr=sr, n_mels=self.mel_n_mels)  # (n_mels, T)
            return mel.astype(np.float32)
        except Exception as e:
            print(f"Error processing log-mel for {audio_path}: {e}")
            return np.zeros((self.mel_n_mels, 1), dtype=np.float32)

    def _get_feature_dim_for_task(self, task: str) -> int:
        """
        task: 'reg' or 'cls'
        Returns feature vector dim based on FEATURE_USAGE toggles.
        """
        if task not in ("reg", "cls"):
            raise ValueError(f"Invalid task '{task}'. Must be 'reg' or 'cls'.")

        use_clap = int(self.feature_usage.get(f"{task}_use_clap", 1)) == 1
        use_mel  = int(self.feature_usage.get(f"{task}_use_mel", 1)) == 1

        dim = 0
        if use_clap:
            dim += self.clap_dim
        if use_mel:
            dim += self.mel_out_dim

        if dim == 0:
            raise ValueError(
                f"[CONFIG ERROR] {task}: both disabled "
                f"({task}_use_clap={self.feature_usage.get(f'{task}_use_clap')}, "
                f"{task}_use_mel={self.feature_usage.get(f'{task}_use_mel')})."
            )
        
        return dim

    def _fuse_np(self, clap_vec: np.ndarray, mel_vec: np.ndarray, task: str) -> np.ndarray:
        """
        Fuse numpy vectors based on feature toggles for the given task.
        Returns a 1D float32 vector.
        """
        if task not in ("reg", "cls"):
            raise ValueError(f"Invalid task '{task}'. Must be 'reg' or 'cls'.")

        use_clap = (self.feature_usage.get(f"{task}_use_clap", 1) == 1)
        use_mel  = (self.feature_usage.get(f"{task}_use_mel", 1) == 1)

        parts = []

        if use_clap:
            v = clap_vec.astype(np.float32)
            if v.ndim != 1 or v.shape[0] != self.clap_dim:
                raise ValueError(f"{task}: clap_vec shape {v.shape} != ({self.clap_dim},)")
            parts.append(v)

        if use_mel:
            v = mel_vec.astype(np.float32)
            if v.ndim != 1 or v.shape[0] != self.mel_out_dim:
                raise ValueError(f"{task}: mel_vec shape {v.shape} != ({self.mel_out_dim},)")
            parts.append(v)

        if len(parts) == 0:
            raise ValueError(f"[CONFIG ERROR] {task}: both CLAP and MEL are disabled.")

        if len(parts) == 1:
            return parts[0]

        return np.concatenate(parts, axis=0)


    def _fuse_torch(self, clap_t: torch.Tensor, mel_t: torch.Tensor, task: str) -> torch.Tensor:
        """
        Fuse torch tensors based on feature toggles for the given task.
        clap_t: (B, clap_dim)
        mel_t : (B, mel_out_dim)
        """
        if task not in ("reg", "cls"):
            raise ValueError(f"Invalid task '{task}'. Must be 'reg' or 'cls'.")

        use_clap = (self.feature_usage.get(f"{task}_use_clap", 1) == 1)
        use_mel  = (self.feature_usage.get(f"{task}_use_mel", 1) == 1)

        parts = []

        if use_clap:
            if clap_t.ndim != 2 or clap_t.shape[1] != self.clap_dim:
                raise ValueError(f"{task}: clap_t shape {tuple(clap_t.shape)} != (B, {self.clap_dim})")
            parts.append(clap_t)

        if use_mel:
            if mel_t.ndim != 2 or mel_t.shape[1] != self.mel_out_dim:
                raise ValueError(f"{task}: mel_t shape {tuple(mel_t.shape)} != (B, {self.mel_out_dim})")
            parts.append(mel_t)

        if len(parts) == 0:
            raise ValueError(f"[CONFIG ERROR] {task}: both CLAP and MEL are disabled.")

        if len(parts) == 1:
            return parts[0]

        # Ensure same device/dtype before cat (defensive)
        if parts[0].device != parts[1].device:
            raise ValueError(f"{task}: feature tensors are on different devices ({parts[0].device} vs {parts[1].device})")
        if parts[0].dtype != parts[1].dtype:
            # allow implicit cast? better to fail loudly to avoid silent precision bugs
            raise ValueError(f"{task}: feature tensors have different dtypes ({parts[0].dtype} vs {parts[1].dtype})")

        return torch.cat(parts, dim=1)

    
    def _mel_spec_to_embedding(self, mel_spec: np.ndarray):
        """
        Run a (n_mels, T) log-mel spectrogram through the MelCRNNEncoder
        and return a 1D embedding of length self.mel_out_dim.
        """
        try:
            if not isinstance(mel_spec, np.ndarray):
                mel_spec = np.array(mel_spec, dtype=np.float32)
            mel_tensor = (
                torch.from_numpy(mel_spec)
                .float()
                .unsqueeze(0)  # add batch dim -> (1, n_mels, T)
                .unsqueeze(0)  # add channel dim -> (1, 1, n_mels, T)
                .to(self.device)
            )
            with torch.no_grad():
                mel_embed = self.mel_encoder(mel_tensor)  # (1, mel_out_dim)

            mel_embed = mel_embed.cpu().numpy().flatten()

            # --- HARD DIMENSION GUARANTEE (Issue 3b) ---
            if mel_embed.shape[0] != self.mel_out_dim:
                print(
                    f"[WARN] _mel_spec_to_embedding produced dim={mel_embed.shape[0]} "
                    f"but expected mel_out_dim={self.mel_out_dim}. Returning zeros."
                )
                return np.zeros(self.mel_out_dim, dtype=np.float32)

            return mel_embed

        except Exception as e:
            print(f"Error converting mel spectrogram to embedding: {e}")
            return np.zeros(self.mel_out_dim, dtype=np.float32)

    def _preprocess_dataset(self, df, dataset_name="unknown", thresholds=None, save_dir=None):
        """
        Applies Ghost Parameter cleaning.
        Uses self.DEFAULT_CLEANING_THRESHOLDS unless overrides are passed.
        Respects 'clean_data' toggle in self.dataset_toggles.
        """
        # --- BYPASS CHECK ---
        # If toggle is 0, skip cleaning entirely and return original data
        if self.dataset_toggles.get("clean_data", 1) == 0:
            print(f"[PREPROCESSING] Skipped for {dataset_name} (Toggle 'clean_data' is 0)")
            return df
        # --------------------
        # Use Central Config if no specific thresholds provided
        if thresholds is None:
            thresholds = self.DEFAULT_CLEANING_THRESHOLDS

        # Build final threshold map (fallback to 0.05 if a key is missing in config)
        final_thresholds = {}
        # We iterate over the families we know exist in the code logic
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
            
            # Get specific threshold for this family
            thresh = final_thresholds[master]
            
            # Find ghosts
            mask = df_clean[master] < thresh
            count = mask.sum()
            
            if count > 0:
                print(f"   -> {master}: Cleaning {count} rows where value < {thresh}")
                
                # 1. Clean Dependents (Set children to 0.0)
                for child in dependents:
                    if child in df_clean.columns:
                        df_clean.loc[mask, child] = 0.0
                
                # 2. Clean Master (NEW: Hard Clamp master to 0.0)
                df_clean.loc[mask, master] = 0.0
                
                total_modified += count
                        
        if total_modified == 0:
            print("   -> No ghost parameters found.")
        else:
            print(f"   -> Done. Total ghost rows cleaned in {dataset_name}: {total_modified}")

            # Save processed copy
            try:
                # --- FIX NAME LOGIC ---
                # Clean the name (remove .csv if present)
                clean_name = dataset_name[:-4] if dataset_name.endswith(".csv") else dataset_name
                
                # Construct save name: prevent double "index_"
                if clean_name.startswith("index_"):
                    save_name = f"{clean_name}_processed.csv"
                else:
                    save_name = f"index_{clean_name}_processed.csv"
                
                # --- FIX SAVE LOCATION LOGIC ---
                # Use provided save_dir, or fallback to default index_dir
                target_dir = save_dir if save_dir is not None else self.index_dir
                
                save_path = os.path.join(target_dir, save_name)
                df_clean.to_csv(save_path, index=False)
                print(f"   -> Saved processed copy to: {save_path}")
            except Exception as e:
                print(f"   [WARN] Could not save processed csv: {e}")
                
        return df_clean

    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} samples with {len(self.df['scenario'].unique())} tone types")
        return self.df
    
    def extract_audio_features(self, audio_path):
        """Extract CLAP embeddings from audio file (WITH PADDING)"""
        try:
            # UPDATED: Use new padding loader
            audio, sr = self._load_and_pad_audio(audio_path, target_sr=48000)
            
            # Get CLAP audio embedding
            inputs = self.clap_processor(audios=audio, return_tensors="pt", sampling_rate=48000)
            inputs = {k: v.to(self.device) for k, v in inputs.items()} 

            with torch.no_grad():
                audio_embed = self.clap_model.get_audio_features(**inputs)

            # --- R1: One-time CLAP shape sanity print + enforce 1D (clap_dim) ---
            vec = audio_embed.detach().cpu().numpy()

            if not hasattr(self, "_did_print_clap_shape"):
                self._did_print_clap_shape = True
                try:
                    print(f"[CLAP] raw tensor shape = {tuple(audio_embed.shape)}, np shape = {vec.shape}")
                except Exception:
                    print("[CLAP] raw tensor shape print failed")

            vec = vec.reshape(-1).astype(np.float32)

            # Enforce exact length (prevents weird (1,512)/(512,1)/extra-dim edge cases)
            if vec.shape[0] != self.clap_dim:
                if not hasattr(self, "_did_warn_clap_dim"):
                    self._did_warn_clap_dim = True
                    print(f"[WARN] CLAP dim mismatch: got {vec.shape[0]} expected {self.clap_dim}. Will pad/truncate.")
                if vec.shape[0] < self.clap_dim:
                    vec = np.pad(vec, (0, self.clap_dim - vec.shape[0]), mode="constant")
                else:
                    vec = vec[:self.clap_dim]

            return vec
        
        
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros(512)
                            
    def _extract_logmel(self, audio, sr=48000, n_mels=96):
        """
        Phase 1.5: Log-mel with fixed width enforcement.
        Uses self.mel_time_steps (calculated from audio_length) as the target.
        """
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=2048,
            hop_length=self.mel_hop_length,
            n_mels=n_mels,
            power=2.0
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = S_db - S_db.mean()
        
        # --- DYNAMIC SAFETY LOCK ---
        # Ensure exact dimensions matching the calculated requirement
        TARGET_WIDTH = self.mel_time_steps
        current_width = S_db.shape[1]
        
        if current_width < TARGET_WIDTH:
            # Pad right side with zeros (Safety catch for rounding errors)
            pad_amt = TARGET_WIDTH - current_width
            S_db = np.pad(S_db, ((0, 0), (0, pad_amt)), mode='constant')
        elif current_width > TARGET_WIDTH:
            # Crop right side
            S_db = S_db[:, :TARGET_WIDTH]
            
        return S_db  # Shape is GUARANTEED (n_mels, TARGET_WIDTH)
            
    def extract_mel_cnn_features(self, audio_path):
        """
        Phase 2: build log-mel and run through the CRNN (CNN → BiLSTM) to get a 128-D vector.
        Updated to use the 2D spectrogram extraction logic for consistency.
        """
        try:
            mel_spec = self._extract_logmel_spectrogram(audio_path)  # (n_mels, T)
            return self._mel_spec_to_embedding(mel_spec)
        except Exception as e:
            print(f"Error processing mel for {audio_path}: {e}")
            return np.zeros(self.mel_out_dim, dtype=np.float32)
        
    
    def extract_text_features(self, text_list):
        """Extract CLAP embeddings from text descriptions"""
        inputs = self.clap_processor(text=text_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # NEW

        with torch.no_grad():
            text_embed = self.clap_model.get_text_features(**inputs)


        return text_embed.cpu().numpy()
    

    def _load_validation_external(self, val_path, max_samples=None):
        """Helper to load validation data from a separate directory structure (Adapted for CRNN)"""
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
            
            # [DATA CLEANING] Preprocess Validation Data (Uses Central Config)
            self.df = self._preprocess_dataset(self.df, dataset_name="validate")
            
            print(f"[INFO] Validation index loaded. Samples: {len(self.df)}")

            X_val, y_tone_val, y_knob_val, X_paths_val = self.prepare_training_data(
                max_samples_per_class=max_samples,
                dataset_key="validate",
                
            )
            return X_val, y_tone_val, y_knob_val, X_paths_val
            
        finally:
            self.index_dir = old_index
            self.audio_root = old_audio
            self.labels_root = old_labels
            self.df = old_df

    def prepare_training_data(self, max_samples_per_class=None, dataset_key="train"):
        """
        Prepare training data from self.df.
        Implements E2E caching: Saves raw Mel and Static (CLAP+DI) features to .npz.
        Returns paths and features for training.
        """
        print(f"Preparing training data for dataset '{dataset_key}'...")

        # --- STATS RESET FOR THIS DATASET ---
        self._stat_padded = 0
        self._stat_cropped = 0
        self._stat_total_processed = 0

        if max_samples_per_class:
            df_sampled = self.df.groupby("scenario").head(max_samples_per_class)
        else:
            df_sampled = self.df

        audio_paths = []
        tone_labels = []
        knob_values = []
        pre_extracted_features = []
        
        count = 0

        for idx, row in df_sampled.iterrows():
            count += 1
            print(f"[{dataset_key}] Processing {count}/{len(df_sampled)}: {row['scenario']}")
            
            audio_path, label_path = self._resolve_sample_paths(dataset_key, row)
            
            if audio_path and os.path.exists(audio_path):
                audio_name = os.path.basename(audio_path)
                cache_path, ds_cache_dir = self._get_cache_path(dataset_key, audio_name)
                
                features = None
                
                # 1. Try to load cache
                if os.path.exists(cache_path):
                    try:
                        data = np.load(cache_path)
                        
                        # Load only what is needed for REGRESSION
                        static_feat = None
                        mel_spec = None
                        mel_embed = None

                        if self.feature_usage.get("reg_use_clap", 1) == 1:
                            if "static_feat" not in data:
                                raise KeyError("Cache missing static_feat but reg_use_clap=1.")
                            static_feat = data["static_feat"]

                        if self.feature_usage.get("reg_use_mel", 1) == 1:
                            if "mel_feat" not in data:
                                raise KeyError("Cache missing mel_feat but reg_use_mel=1.")
                            mel_spec = data["mel_feat"]

                            # --- SHAPE CHECK ---
                            if mel_spec.ndim != 2:
                                raise ValueError(f"Invalid cached mel_feat shape {mel_spec.shape}")

                            mel_embed = self._mel_spec_to_embedding(mel_spec)

                        # Build fused regression features
                        if self.feature_usage.get("reg_use_clap", 1) == 1 and static_feat is None:
                            raise ValueError("reg_use_clap=1 but static_feat is None (cache issue).")
                        if self.feature_usage.get("reg_use_mel", 1) == 1 and mel_embed is None:
                            raise ValueError("reg_use_mel=1 but mel_embed is None (cache issue).")

                        if self.feature_usage.get("reg_use_clap", 1) == 1 and self.feature_usage.get("reg_use_mel", 1) == 1:
                            features = np.concatenate([static_feat, mel_embed], axis=0)
                        elif self.feature_usage.get("reg_use_clap", 1) == 1:
                            features = static_feat.astype(np.float32)
                        else:
                            features = mel_embed.astype(np.float32)

                        # Cache loaded successfully
                        pre_extracted_features.append(features)
                        audio_paths.append(audio_path)
                        tone_labels.append(row["scenario"])
                        knob_vals = [row[f"n_{param}"] for param in self.knob_params]
                        knob_values.append(knob_vals)
                        continue # Fast path

                    except Exception as e:
                        print(f"[WARN] Failed to load cache: {cache_path}. Error: {e}")
                        try:
                            os.remove(cache_path)
                        except Exception:
                            pass

                # 2. Extract and Save (SLOW PATH)
                print(f"[EXTRACTING] Cache missing or invalid for {row['scenario']}...")

                # 2a. Extract only what REGRESSION needs
                static_feat = None
                mel_spec = None
                mel_embed = None

                if self.feature_usage.get("reg_use_clap", 1) == 1:
                    # Calls extract_audio_features -> _load_and_pad_audio
                    clap_feat = self.extract_audio_features(audio_path)
                    static_feat = clap_feat.astype(np.float32)

                if self.feature_usage.get("reg_use_mel", 1) == 1:
                    mel_spec = self._extract_logmel_spectrogram(audio_path)  # Raw (n_mels, T)
                    mel_embed = self._mel_spec_to_embedding(mel_spec)        # Embedding

                # Build fused regression features
                if self.feature_usage.get("reg_use_clap", 1) == 1 and self.feature_usage.get("reg_use_mel", 1) == 1:
                    features = np.concatenate([static_feat, mel_embed], axis=0)
                elif self.feature_usage.get("reg_use_clap", 1) == 1:
                    features = static_feat
                elif self.feature_usage.get("reg_use_mel", 1) == 1:
                    features = mel_embed.astype(np.float32)
                else:
                    raise ValueError("Invalid FEATURE_USAGE: both reg_use_clap=0 and reg_use_mel=0.")

                # Save to cache (only keys that were actually computed)
                try:
                    os.makedirs(ds_cache_dir, exist_ok=True)
                    save_kwargs = {}
                    if static_feat is not None:
                        save_kwargs["static_feat"] = static_feat
                    if mel_spec is not None:
                        save_kwargs["mel_feat"] = mel_spec
                    np.savez_compressed(cache_path, **save_kwargs)
                
                
                except Exception as e:
                    print(f"[ERR] Failed to save cache to {cache_path}: {e}")

                pre_extracted_features.append(features)
                audio_paths.append(audio_path)
                tone_labels.append(row["scenario"])
                knob_vals = [row[f"n_{param}"] for param in self.knob_params]
                knob_values.append(knob_vals)

            else:
                print(f"[WARN] audio not found: {audio_path}")

        # --- STATS REPORTING ---
        print(f"\n[AUDIO STATS] Dataset '{dataset_key}':")
        print(f"   -> Audio Length Target: {self.audio_length} seconds")
        print(f"   -> Files Zero-Padded:   {self._stat_padded}")
        print(f"   -> Files Cropped:       {self._stat_cropped}")
        print(f"   -> Total Processed:     {count}\n")

        return (
            np.array(pre_extracted_features), 
            np.array(tone_labels), 
            np.array(knob_values),
            np.array(audio_paths) # New return
        )
        
    
    
    def _pre_load_ram_cache(self, X_paths, y_knobs, dataset_name="train"):
        """
        Loads cache files (.npz) into memory lists to avoid Disk I/O during training.
        Returns: list of (mel_tensor, static_tensor, label_tensor)
        """
        import sys
        
        data_list = []
        total = len(X_paths)
        print(f"[OPTIMIZATION] Pre-loading {total} {dataset_name} cache files to RAM...")
        
        # Rough estimate of RAM usage (assuming ~150KB per sample for tensors)
        est_ram = total * 0.15 
        print(f"   (approx RAM usage: {est_ram:.1f} MB)")
        
        for i, (path, label) in enumerate(zip(X_paths, y_knobs)):
            if i % 1000 == 0 and i > 0:
                print(f"   -> Loading {i}/{total}...")
                
            try:
                # Reuse the logic from EndToEndDataset regarding path resolution
                audio_name = os.path.basename(path)
                path_parts = path.split(os.sep)
                try:
                    audio_root_index = path_parts.index(os.path.basename(self.audio_root))
                    # If loading train data, key is usually 'train' etc.
                    # We try to infer or fallback to dataset_name provided
                    key_in_path = path_parts[audio_root_index + 1]
                    ds_key = key_in_path
                except ValueError:
                    ds_key = dataset_name # fallback
                
                cache_path, _ = self._get_cache_path(ds_key, audio_name)
                
                if os.path.exists(cache_path):
                    data = np.load(cache_path)

                    # Respect regression feature toggles
                    use_mel  = (self.feature_usage.get("reg_use_mel", 1) == 1)
                    use_clap = (self.feature_usage.get("reg_use_clap", 1) == 1)

                    # --- Mel (only if enabled) ---
                    target_width = getattr(self, "mel_time_steps", 32)
                    if use_mel:
                        if "mel_feat" not in data:
                            raise KeyError(f"Cache missing mel_feat but reg_use_mel=1: {cache_path}")
                        mel_spec = data["mel_feat"]

                        # SHAPE CHECK: MUST BE 2D SPECTROGRAM
                        if mel_spec.ndim != 2:
                            raise ValueError(
                                f"Invalid cached mel_feat shape {mel_spec.shape} in {cache_path}; "
                                f"expected 2D (n_mels, T)."
                            )

                        mel_t = torch.from_numpy(mel_spec).float().unsqueeze(0)  # (1, n_mels, T)
                    else:
                        # Safe zero mel tensor (won't be used downstream when reg_use_mel=0)
                        mel_t = torch.zeros(1, self.mel_n_mels, target_width)

                    # --- CLAP/static (only if enabled) ---
                    clap_dim = getattr(self, "clap_dim", 512)
                    if use_clap:
                        if "static_feat" not in data:
                            raise KeyError(f"Cache missing static_feat but reg_use_clap=1: {cache_path}")
                        static_t = torch.FloatTensor(data["static_feat"])
                    else:
                        # Safe zero static tensor (won't be used downstream when reg_use_clap=0)
                        static_t = torch.zeros(clap_dim)

                    label_t = torch.FloatTensor(label)
                    data_list.append((mel_t, static_t, label_t))

                else:
                    # Should ideally not happen if prepare_training_data ran first
                    # Create dummy zeros (respect feature toggles)
                    # print(f"[WARN] RAM Cache miss: {cache_path}")

                    use_mel  = (self.feature_usage.get("reg_use_mel", 1) == 1)
                    use_clap = (self.feature_usage.get("reg_use_clap", 1) == 1)

                    # mel tensor (only needed if reg_use_mel=1, otherwise safe zeros)
                    target_width = getattr(self, "mel_time_steps", 32)
                    if use_mel:
                        mel_t = torch.zeros(1, self.mel_n_mels, target_width)
                    else:
                        mel_t = torch.zeros(1, self.mel_n_mels, target_width)

                    # static tensor (only needed if reg_use_clap=1, otherwise safe zeros)
                    clap_dim = getattr(self, "clap_dim", 512)
                    if use_clap:
                        static_t = torch.zeros(clap_dim)
                    else:
                        static_t = torch.zeros(clap_dim)

                    label_t = torch.zeros(len(self.knob_params))
                    data_list.append((mel_t, static_t, label_t))
                    
            except Exception as e:
                print(f"[ERR] RAM Load error {path}: {e}")
                
        print(f"   -> Loading {total}/{total}... Done.")
        return data_list
    
    def _rebuild_rf_features_from_paths(self, X_paths, y_tone_raw, default_dataset_name="train"):
        """
        Rebuild fused [static_feat | mel_embed] feature matrix for Random Forest
        using the *current* mel_encoder weights and cached static/mel features.

        This is used AFTER end-to-end training so that:
          - RF sees embeddings from the final trained mel_encoder (no mismatch).
        """
        features = []
        labels = []
        total = len(X_paths)
        print(f"[Classifier-Align] Rebuilding features for {total} samples (default dataset='{default_dataset_name}')...")

        for i, (path, label) in enumerate(zip(X_paths, y_tone_raw)):
            if i > 0 and i % 1000 == 0:
                print(f"[Classifier-Align]   -> Processed {i}/{total}...")

            try:
                audio_name = os.path.basename(path)
                path_parts = path.split(os.sep)

                # Try to infer dataset key from path (like _pre_load_ram_cache)
                try:
                    audio_root_name = os.path.basename(self.audio_root)
                    audio_root_index = path_parts.index(audio_root_name)
                    key_in_path = path_parts[audio_root_index + 1]
                    ds_key = key_in_path
                except ValueError:
                    ds_key = default_dataset_name

                cache_path, _ = self._get_cache_path(ds_key, audio_name)

                static_feat = None
                mel_spec = None

                if os.path.exists(cache_path):
                    data = np.load(cache_path)

                    # Load what we can from cache; if a required key is missing,
                    # recompute ONLY that part (keeps cls feature toggles independent of reg caching).
                    
                    if self.feature_usage.get("cls_use_clap", 1) == 1:
                        if "static_feat" in data:
                            static_feat = data["static_feat"]
                        else:
                            print(f"[Classifier-Align WARN] Cache missing static_feat for {path}, recomputing CLAP.")
                            static_feat = self.extract_audio_features(path)

                    if self.feature_usage.get("cls_use_mel", 1) == 1:
                        if "mel_feat" in data:
                            mel_spec = data["mel_feat"]
                        else:
                            print(f"[Classifier-Align WARN] Cache missing mel_feat for {path}, recomputing mel.")
                            mel_spec = self._extract_logmel_spectrogram(path)


                else:
                    # Fallback: recompute only what CLASSIFICATION needs
                    print(f"[Classifier-Align WARN] Cache missing for {path}, recomputing features from audio.")

                    if self.feature_usage.get("cls_use_clap", 1) == 1:

                        static_feat = self.extract_audio_features(path)

                    if self.feature_usage.get("cls_use_mel", 1) == 1:
                        mel_spec = self._extract_logmel_spectrogram(path)

                # Safety: mel_spec must be 2D (only if mel is enabled for classification)
                if self.feature_usage.get("cls_use_mel", 1) == 1:
                    if mel_spec is None:
                        raise ValueError(f"cls_use_mel=1 but mel_spec is None for {cache_path}")
                    if mel_spec.ndim != 2:
                        raise ValueError(f"Invalid mel_feat shape {mel_spec.shape} for {cache_path}")

                # IMPORTANT: mel_embed is computed with the *current* mel_encoder
                mel_embed = None
                if self.feature_usage.get("cls_use_mel", 1) == 1:

                    mel_embed = self._mel_spec_to_embedding(mel_spec)

                if self.feature_usage.get("cls_use_clap", 1) == 1 and self.feature_usage.get("cls_use_mel", 1) == 1:
                    fused_feat = np.concatenate([static_feat, mel_embed], axis=0)
                elif self.feature_usage.get("cls_use_clap", 1) == 1:
                    fused_feat = static_feat.astype(np.float32)
                elif self.feature_usage.get("cls_use_mel", 1) == 1:
                    fused_feat = mel_embed.astype(np.float32)
                else:
                    raise ValueError("Invalid FEATURE_USAGE: both cls_use_clap=0 and cls_use_mel=0.")



                features.append(fused_feat)
                labels.append(label)

            except Exception as e:
                print(f"[Classifier-Align ERR] Failed to rebuild features for {path}: {e}")

        features = np.array(features)
        labels = np.array(labels)
        print(f"[Classifier-Align] Done. Built feature matrix of shape: {features.shape}")
        return features, labels

    def train_models(
        self,
        max_samples_per_class=20,
        epochs=300,
        batch_size=32,
        learning_rate=0.001,
        dataset_toggles: dict = None,
        validate_data_dir=r"C:\Aarush\Project\Clap\data_validate"
    ):
        """
        Train knob regressor end-to-end first (mel_encoder + MLP),
        then align and train the Random Forest classifier using the
        *final* mel_encoder for feature extraction (Option B).
        """
        print("Training models...")

        # Initialize tiling counter for this run
        self._tiled_count = 0

        # timing dict for this run
        start_all = time.time()
        self.timing_info = {
            "start_all": start_all
        }

        # Merge external toggles
        if dataset_toggles is not None:
            self.dataset_toggles.update(dataset_toggles)

        print(f"[INFO] Phase 3 dataset toggles: {self.dataset_toggles}")

        dataset_map = self._discover_datasets()
        print(f"[INFO] Discovered datasets: {list(dataset_map.keys())}")

        # ---- 1) load MAIN train df ----
        if "train" in dataset_map:
            main_df = self._load_index_csv(dataset_map["train"])
        else:
            main_df = self.load_data()

        # [DATA CLEANING] Preprocess Main Dataset (Uses Central Config)
        main_df = self._preprocess_dataset(main_df, dataset_name="train")

        all_X = []
        all_y_tones = []
        all_y_knobs = []
        all_X_paths = []

        # ---- a) always include train ----
        self.df = main_df
        X_train_main, y_tones_main, y_knobs_main, X_paths_main = self.prepare_training_data(
            max_samples_per_class=max_samples_per_class,
            dataset_key="train",
            
        )
        all_X.append(X_train_main)
        all_y_tones.append(y_tones_main)
        all_y_knobs.append(y_knobs_main)
        all_X_paths.append(X_paths_main)
        print(f"[INFO] train: {len(X_train_main)} samples prepared.")

        # timing: base dataset done
        self.timing_info["base_done"] = time.time()

        # ---- b) include all other normal datasets (auto) ----
        for ds_key, ds_path in dataset_map.items():
            if ds_key == "train":
                continue

            toggle_name = f"{ds_key}_aware"
            if self.dataset_toggles.get(toggle_name, 1) != 1:
                print(f"[INFO] dataset '{ds_key}' present but {toggle_name}=0 → skipping.")
                continue

            ds_df = self._load_index_csv(ds_path)

            # [DATA CLEANING] Preprocess Additional Datasets (Uses Central Config)
            ds_df = self._preprocess_dataset(ds_df, dataset_name=ds_key)

            self.df = ds_df

            X_ds, y_ds_tones, y_ds_knobs, X_paths_ds = self.prepare_training_data(
                max_samples_per_class=max_samples_per_class,
                dataset_key=ds_key,
                
            )
            if len(X_ds) > 0:
                all_X.append(X_ds)
                all_y_tones.append(y_ds_tones)
                all_y_knobs.append(y_ds_knobs)
                all_X_paths.append(X_paths_ds)
                print(f"[INFO] {ds_key}: {len(X_ds)} samples prepared and merged into main training pool.")

                if ds_key == "temporal":
                    self.timing_info["temporal_done"] = time.time()
            else:
                print(f"[WARN] dataset '{ds_key}' enabled but produced 0 samples.")

        # restore main df for later use
        self.df = main_df

        # [AUDIO TILING REPORT - TRAINING]
        print(f"\n[AUDIO TILING] Training Data: {self._tiled_count} audio files were short and tiled to {self.audio_length}s.")

        # --- RESET TILING COUNTER FOR VALIDATION ---
        self._tiled_count = 0

        # ---- c) merge all supervised rows (TRAINING POOL) ----
        X_train = np.concatenate(all_X, axis=0)
        y_tone_train_raw = np.concatenate(all_y_tones, axis=0)
        y_knob_train = np.concatenate(all_y_knobs, axis=0)
        X_paths_train = np.concatenate(all_X_paths, axis=0)
        print(f"[INFO] merged supervised datasets → {X_train.shape[0]} total TRAINING samples.")

        if len(X_train) == 0:
            raise ValueError("No training data available")

        # remember feature dim for regression (static + mel_embed)
        self.feature_dim = X_train.shape[1]

        # ---- d) Load EXTERNAL VALIDATION data (for regression + later RF) ----
        print(f"[INFO] Loading external validation data...")
        X_val_dummy, y_tone_val_raw, y_knob_val, X_paths_val = self._load_validation_external(
            validate_data_dir,
            max_samples=max_samples_per_class
        )

        print(f"[AUDIO TILING] Validation Data: {self._tiled_count} audio files were short and tiled to {self.audio_length}s.")
        print(f"[INFO] External validation loaded → {X_val_dummy.shape[0]} total VALIDATION samples.")

        if len(X_val_dummy) == 0:
            raise ValueError("No validation data available. Check path or index_validate.csv")

        # NOTE:
        # X_train / X_val_dummy here are *pre-E2E* fused features (old mel_encoder).
        # We ONLY use them to:
        #   - fit knob_scaler (inside _train_knob_neural_network)
        # y_* and X_paths_* drive regression and RAM loading.
        # RF training will be done LATER using the final mel_encoder state.

        # ---- Save raw tone labels & paths for later RF alignment ----
        self._rf_y_tone_train_raw = y_tone_train_raw
        self._rf_y_tone_val_raw = y_tone_val_raw
        self._rf_X_paths_train = X_paths_train
        self._rf_X_paths_val = X_paths_val

        
                
        # ---- 4) TRAIN E2E NN (REGRESSION) - UNCHANGED WORKFLOW ----
        print("Training neural network for knob parameters...")
        self.timing_info["train_start"] = time.time()

        # [RAM CACHING] Pre-load data to RAM (uses cached static_feat + mel_feat)
        train_ram_data = self._pre_load_ram_cache(X_paths_train, y_knob_train, dataset_name="train")
        val_ram_data = self._pre_load_ram_cache(X_paths_val, y_knob_val, dataset_name="validate")
        print(f"[OPTIMIZATION] Loaded {len(train_ram_data)} training and {len(val_ram_data)} val samples to RAM.")

        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] STARTING E2E TRAINING (RAM Optimized)...")

        # IMPORTANT: regression workflow unchanged
        self._train_knob_neural_network(
            X_train,          # Only for scaler fit
            train_ram_data,   # RAM data list
            val_ram_data,     # RAM data list
            y_knob_train,
            y_knob_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_loss_weights=1,
                       
            
        )

        # ---- 5) AFTER E2E: ALIGN CLASSIFIER WITH FINAL mel_encoder (Option B) ----
        print("\n[Classifier-Align] Training tone classifier on FINAL mel encoder embeddings...")

        # Rebuild features for TRAIN and VALIDATION using the *trained* mel_encoder

        X_train_rf, y_tone_train_rf = self._rebuild_rf_features_from_paths(
            self._rf_X_paths_train,
            self._rf_y_tone_train_raw,
            default_dataset_name="train"
        )
        X_val_rf, y_tone_val_rf = self._rebuild_rf_features_from_paths(
            self._rf_X_paths_val,
            self._rf_y_tone_val_raw,
            default_dataset_name="validate"
        )

        # Store classification feature width separately.
        # Do NOT overwrite regression feature_dim, because reg_use_* and cls_use_* can differ.
        self.cls_feature_dim = X_train_rf.shape[1]

        # Encode labels for RF
        y_tone_train = self.label_encoder.fit_transform(y_tone_train_rf)
        try:
            y_tone_val = self.label_encoder.transform(y_tone_val_rf)
        except ValueError:
            print("[WARN] Validation set contains labels not seen in training! Falling back to fitting on combined.")
            combined_labels = np.concatenate([y_tone_train_rf, y_tone_val_rf])
            self.label_encoder.fit(combined_labels)
            y_tone_train = self.label_encoder.transform(y_tone_train_rf)
            y_tone_val = self.label_encoder.transform(y_tone_val_rf)

        # Save for later report usage (validation report in _train_knob_neural_network)
        self.X_tone_test = X_val_rf
        self.y_tone_test = y_tone_val

        if self.classifier_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self.tone_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.tone_classifier.fit(X_train_rf, y_tone_train)
            y_pred = self.tone_classifier.predict(X_val_rf)

            print("Tone Classification Report (aligned with final mel encoder):")
            print(classification_report(y_tone_val, y_pred, target_names=self.label_encoder.classes_))

            self.tone_classifier_nn = None

        elif self.classifier_type == "nn_head":
            # Train a lightweight vector NN head on the SAME fused vector RF uses
            self.tone_classifier = None

            num_classes = len(self.label_encoder.classes_)
            input_dim = X_train_rf.shape[1]
            self.cls_feature_dim = input_dim

            self.tone_classifier_nn = ToneClassifierHead(input_dim=input_dim, num_classes=num_classes).to(self.device)
            self.tone_classifier_nn.train()

            Xtr = torch.from_numpy(X_train_rf).float().to(self.device)
            ytr = torch.from_numpy(y_tone_train).long().to(self.device)
            Xva = torch.from_numpy(X_val_rf).float().to(self.device)
            yva = torch.from_numpy(y_tone_val).long().to(self.device)

            # Linked to main learning_rate argument
            opt = torch.optim.Adam(self.tone_classifier_nn.parameters(), lr=learning_rate)
            crit = nn.CrossEntropyLoss()

            # Early stopping setup
            best_val_loss = float('inf')
            patience_counter = 0
            best_nn_state = None
            
            print(f"[NN_HEAD] Starting training for {epochs} epochs on {self.device}...")

            for epoch in range(epochs):
                epoch_start = time.time()
                
                # Training step
                self.tone_classifier_nn.train()
                opt.zero_grad()
                logits = self.tone_classifier_nn(Xtr)
                loss = crit(logits, ytr)
                loss.backward()
                opt.step()
                
                train_loss = loss.item()

                # Validation step
                self.tone_classifier_nn.eval()
                with torch.no_grad():
                    v_logits = self.tone_classifier_nn(Xva)
                    v_loss = crit(v_logits, yva).item()

                # Early stopping logic (matches Regressor logic)
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    patience_counter = 0
                    best_nn_state = copy.deepcopy(self.tone_classifier_nn.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= 30:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

                # Logging (Matches Regressor format: [HH:MM:SS] Epoch X/Y (dur): Train=..., Val=...)
                if (epoch + 1) % 1 == 0:
                    current_time = datetime.now().strftime("[%H:%M:%S]")
                    epoch_dur = time.time() - epoch_start
                    print(f"{current_time} Epoch {epoch+1}/{epochs} ({epoch_dur:.3f}s): "
                          f"Train Loss = {train_loss:.6f}, Val Loss = {v_loss:.6f}")

            # Restore best weights
            if best_nn_state is not None:
                self.tone_classifier_nn.load_state_dict(best_nn_state)
                print(f"[INFO] Restored best NN_HEAD weights (val_loss={best_val_loss:.6f})")

            # Final Evaluation for Report
            self.tone_classifier_nn.eval()
            with torch.no_grad():
                v_logits = self.tone_classifier_nn(Xva)
                v_pred = torch.argmax(v_logits, dim=1).cpu().numpy()

            print("Tone Classification Report (NN-head, aligned with final mel encoder):")
            print(classification_report(y_tone_val, v_pred, target_names=self.label_encoder.classes_))
        else:
            raise ValueError(f"Unknown classifier_type='{self.classifier_type}'. Use 'rf' or 'nn_head'.")


        # ---- 6) text embeddings unchanged ----
        print("Creating text embeddings for tone matching...")
        for tone_type, descriptions in self.tone_descriptions.items():
            text_embeds = self.extract_text_features(descriptions)
            self.tone_embeddings[tone_type] = np.mean(text_embeds, axis=0)
    
    def _train_knob_neural_network(
        self,
        X_train, 
        train_ram_data, # <--- CHANGED NAME/TYPE
        val_ram_data,   # <--- CHANGED NAME/TYPE
        y_knob_train,
        y_knob_test,
        epochs=300, batch_size=32, learning_rate=0.001,
        use_loss_weights: int = 0,
    ):
        """
        Train neural network for knob parameter prediction.
        """

        # Guard invalid regression feature config (must match your finalized requirements)
        if self.feature_usage.get("reg_use_clap", 1) == 0 and self.feature_usage.get("reg_use_mel", 1) == 0:
            raise ValueError("Invalid FEATURE_USAGE: reg_use_clap=0 and reg_use_mel=0 (no regression features).")
        
        # 1. SCALER FIT
        self.knob_scaler.fit(X_train)

        # 2. DATASETS (Using RAM mode)
        # Note: We pass None for y_knob because labels are inside the RAM tuples now
        train_dataset = EndToEndDataset(self, data_source=train_ram_data, mode="ram")
        val_dataset = EndToEndDataset(self, data_source=val_ram_data, mode="ram")
        
        # 3. DATALOADERS with Collate
        # Important: Since we are in RAM, num_workers can be 0 (main process) without lag
        num_workers = 0 
        print(f"[INFO] Using {num_workers} DataLoader worker processes (RAM optimized).")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )
        
        # ... (rest of the function remains mostly identical, just verify `batch_y` logic)
        # In the training loop:
        # for batch_mel, batch_static, batch_y in train_loader:
        # This will work perfectly because RAM dataset returns (mel, static, y) tuples.
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # 4. MODEL INIT (Standard...)
        self.knob_regressor_net = KnobParameterNet(
            input_dim=self.feature_dim,
            hidden_dims=[256, 128, 64],
            output_dim=len(self.knob_params),
            dropout_rate=0.25
        ).to(self.device)

        # ... (Proceed with existing training loop code ...)
        # ... No changes needed inside the loop itself ...
        
        # 5. OPTIMIZER & LOSS
        #main_criterion = nn.L1Loss(reduction='none')
        #main_criterion = nn.HuberLoss(reduction='none', delta=1.0)
        main_criterion = nn.MSELoss(reduction='none')
        # OPTIMIZE BOTH Encoder AND Regressor
        combined_params = []
        if self.feature_usage.get("reg_use_mel", 1) == 1:
            combined_params += list(self.mel_encoder.parameters())
        combined_params += list(self.knob_regressor_net.parameters())


        # AdamW usually prefers a slightly higher weight_decay than Adam (e.g., 1e-2 or 1e-4)
        #optimizer = optim.AdamW(combined_params, lr=learning_rate, weight_decay=1e-2)
        
        optimizer = optim.Adam(combined_params, lr=learning_rate, weight_decay=1e-5)
        #optimizer = optim.Adam(combined_params, lr=learning_rate, weight_decay=1e-3)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

        # Weights setup (per-parameter loss weighting)
        # IMPORTANT: keys MUST match self.knob_params exactly and cover all 17 params.
        # Start with all-1.0 (unweighted), then adjust as needed.
        param_weight_table = {
            # Distortion / Drive
            "overdrivedrive":   1.0,
            "distortiondrive":  1.0,
            "distortiontone":   1.0,

            # EQ
            "eqbass":           1.0,
            "eqmid":            1.0,
            "eqtreble":         1.0,

            # Chorus
            "chorusrate":       1.0,
            "chorusdepth":      1.0,
            "chorusmix":        1.0,

            # Delay
            "delaytime":        1.0,
            "delayfeedback":    1.0,
            "delaymix":         1.0,

            # Reverb
            "reverbt60":        1.0,
            "reverbdamp":       1.0,
            "reverbsize":       1.0,
            "reverbwet":        1.0,

            # Master
            "mastervolume":     1.0,
        }

        # Build weights in the exact knob order used everywhere else
        tuned_weights = torch.tensor(
            [float(param_weight_table[p]) for p in self.knob_params],
            dtype=torch.float32,
            device=self.device
        )


        if use_loss_weights == 1:
            param_weights = tuned_weights
            print("➡️ Using loss weighting.")
        else:
            param_weights = torch.ones(len(self.knob_params), dtype=torch.float32, device=self.device)
            print("➡️ Using UNWEIGHTED main loss.")

        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = -1
        best_regressor_state = None
        best_mel_state = None

        print(f"Training neural network on {self.device}...")
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] STARTING E2E TRAINING...")
        
        epochs_run = int(epochs)

        # History buffers for plotting
        train_loss_history = []
        val_loss_history = []
        
        for epoch in range(epochs):
            
            
            # --- NEW: Start Timer ---
            epoch_start_timestamp = time.time()            
            
            self.knob_regressor_net.train()
            self.mel_encoder.train() # Trainable Encoder
            
            total_loss = 0.0
            num_batches = 0

            # ========== MAIN TRAIN LOOP (E2E) ==========
            for batch_mel, batch_static, batch_y in train_loader:
                batch_mel = batch_mel.to(self.device)
                batch_static = batch_static.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # 1. MelCRNN Forward (Trainable)
                # batch_mel is (B, 1, n_mels, T)
                
                # Optional mel forward (only if enabled)
                if self.feature_usage.get("reg_use_mel", 1) == 1:
                    mel_embed = self.mel_encoder(batch_mel)  # (B, mel_out_dim)
                else:
                    mel_embed = torch.zeros((batch_static.shape[0], self.mel_out_dim),
                                            device=batch_static.device, dtype=batch_static.dtype)

                # Fuse regression features based on toggles
                combined_features = self._fuse_torch(batch_static, mel_embed, task="reg")


                # 3. Knob Net Forward
                # Note: We rely on BatchNorm in the net rather than re-scaling per batch inside the loop
                preds = self.knob_regressor_net(combined_features)
                
                per_param = main_criterion(preds, batch_y)
                weighted = per_param * param_weights
                main_loss = weighted.mean()

                main_loss.backward()
                optimizer.step()

                total_loss += main_loss.item()
                num_batches += 1

            avg_train_loss = total_loss / max(1, num_batches)

            
            
            # ========== VALIDATION ==========
            self.knob_regressor_net.eval()
            self.mel_encoder.eval()
            
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for val_mel, val_static, val_y in val_loader:
                    val_mel = val_mel.to(self.device)
                    val_static = val_static.to(self.device)
                    val_y = val_y.to(self.device)

                    if self.feature_usage.get("reg_use_mel", 1) == 1:
                        val_mel_embed = self.mel_encoder(val_mel)
                    else:
                        val_mel_embed = torch.zeros((val_static.shape[0], self.mel_out_dim),
                                                    device=val_static.device, dtype=val_static.dtype)

                    val_features = self._fuse_torch(val_static, val_mel_embed, task="reg")
                    
                    val_preds = self.knob_regressor_net(val_features)


                    batch_loss = (main_criterion(val_preds, val_y) * param_weights).mean().item()
                    val_loss += batch_loss
                    val_batches += 1
            
            val_loss = val_loss / max(1, val_batches)
            scheduler.step(val_loss)

            # Save history for plotting
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(val_loss)

            # early stopping + track best validation checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0


                best_regressor_state = copy.deepcopy(self.knob_regressor_net.state_dict())
                best_mel_state = copy.deepcopy(self.mel_encoder.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= 30:
                    print(f"Early stopping at epoch {epoch+1}")
                    epochs_run = int(epoch + 1)
                    break

            # Print EVERY 20 epoch
            #if (epoch + 1) % 20 == 0:
            #    print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")

            # Print EVERY epoch
            if (epoch + 1) % 1 == 0: 
                current_time = datetime.now().strftime("[%H:%M:%S]")
                epoch_duration = time.time() - epoch_start_timestamp
                print(f"{current_time} Epoch {epoch+1}/{epochs} ({epoch_duration:.1f}s): Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # ===== load best validation checkpoint before final evaluation =====
        if best_regressor_state is not None and best_mel_state is not None:
            self.knob_regressor_net.load_state_dict(best_regressor_state)
            self.mel_encoder.load_state_dict(best_mel_state)
            print(f"\n[INFO] Restored best validation weights from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
        else:
            print("\n[WARN] No best-validation checkpoint recorded; using last-epoch weights.")

        # ===== final evaluation (Calculate Final Predictions on Val Set) =====
        self.knob_regressor_net.eval()
        self.mel_encoder.eval()
        all_preds = []
        with torch.no_grad():
            for val_mel, val_static, val_y in val_loader:
                
                
                val_mel = val_mel.to(self.device)
                val_static = val_static.to(self.device)
                
                if self.feature_usage.get("reg_use_mel", 1) == 1:
                    val_mel_embed = self.mel_encoder(val_mel)
                else:
                    val_mel_embed = torch.zeros((val_static.shape[0], self.mel_out_dim),
                                                device=val_static.device, dtype=val_static.dtype)

                val_features = self._fuse_torch(val_static, val_mel_embed, task="reg")
                
                val_preds = self.knob_regressor_net(val_features)                
                
                all_preds.append(val_preds.cpu().numpy())
        
        final_predictions = np.concatenate(all_preds, axis=0)

        print("\nParameter-wise Performance:")
        total_mse = 0.0
        per_param_results = []
        for i, param in enumerate(self.knob_params):
            param_mse = mean_squared_error(y_knob_test[:, i], final_predictions[:, i])
            total_mse += param_mse
            per_param_results.append((param, param_mse, float(param_weights[i].item())))
            if use_loss_weights == 1:
                print(f"{param}: MSE = {param_mse:.6f} (weight={param_weights[i].item():.2f})")
            else:
                print(f"{param}: MSE = {param_mse:.6f}")

        avg_mse = total_mse / len(self.knob_params)
        print(f"\nAverage MSE across all parameters: {avg_mse:.6f}")

        overall_r2 = r2_score(y_knob_test, final_predictions)
        print(f"Overall R² Score: {overall_r2:.4f}")

        # write report (same format you wanted)
        os.makedirs("models", exist_ok=True)
        #from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _model = str(REPORT_TUNING.get("model_name", "crnn")).strip().lower() or "crnn"
        report_path = os.path.join("models", f"test_evaluation_results_{timestamp}_val_{_model}.txt")

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

                loss_plot_name = f"epoch_run_{timestamp}_{_model}.png"
                loss_plot_path = os.path.join("models", loss_plot_name)
                plt.savefig(loss_plot_path, dpi=150)
                plt.close()
        except Exception as e:
            print(f"[WARN] Failed to save loss curve plot: {e}")
 
        
        from sklearn.metrics import mean_absolute_error
        # remember last report so save_models() can append timings after saving


        self.last_report_path = report_path
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("================================================================================\n")
            f.write(f"ToneCraft - VALIDATION SET EVALUATION RESULTS - {str(REPORT_TUNING.get('model_name','crnn')).upper()}\n")
            f.write("================================================================================\n\n")
            
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

                # per-class metrics in table form (Phase 3 requirement)
                from sklearn.metrics import classification_report, confusion_matrix
                report_dict = classification_report(
                    y_tone_test,
                    tone_preds,
                    target_names=class_names,
                    output_dict=True
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

                # confusion matrix (rows = true, cols = pred)
                cm = confusion_matrix(y_tone_test, tone_preds, labels=list(range(len(class_names))))
                
                f.write("Confusion Matrix (rows = true, cols = pred):\n")
                # header (wider column width)
                f.write("                " + "".join([f"{c:>10s}" for c in class_names]) + "\n")
                for i, row_cm in enumerate(cm):
                    f.write(f"{class_names[i]:<15s}" + "".join([f"{v:>10d}" for v in row_cm]) + "\n")

                # also write the sklearn text report (for continuity with Phase 2)
                #f.write("\nFull Classification Report (sklearn):\n")
                #f.write(classification_report(y_tone_test, tone_preds, target_names=class_names))
            else:
                f.write("Overall Accuracy: 0.00%\n")
            f.write("\n\n")
            
            f.write("================================================================================\n")
            f.write("PARAMETER PREDICTION PERFORMANCE\n")
            f.write("================================================================================\n\n")
            f.write("Overall Metrics:\n")
            f.write(f"  Mean Squared Error (MSE):  {avg_mse:.6f}\n")
            overall_mae = mean_absolute_error(y_knob_test.reshape(-1), final_predictions.reshape(-1))
            f.write(f"  Mean Absolute Error (MAE): {overall_mae:.6f}\n")
            f.write(f"  R² Score:                  {overall_r2:.4f}\n\n")

            # collect per-parameter metrics in order
            per_param_metrics = []
            for i, (name, mse_val, w_val) in enumerate(per_param_results):
                y_true_i = y_knob_test[:, i]
                y_pred_i = final_predictions[:, i]
                mae_i = mean_absolute_error(y_true_i, y_pred_i)
                try:
                    r2_i = r2_score(y_true_i, y_pred_i)
                except Exception:
                    r2_i = 0.0
                per_param_metrics.append({
                    "idx": i + 1,
                    "name": name,
                    "mse": mse_val,
                    "mae": mae_i,
                    "r2": r2_i,
                    "weight": w_val,
                })

            # table header (with S.No)
            
            if use_loss_weights == 1:
                f.write(f"{'S.No':>4s}  {'Parameter':<24s}{'MSE':>12s}{'MAE':>12s}{'R²':>10s}{'Weight':>10s}\n")
                f.write("-" * 72 + "\n")
            else:
                f.write(f"{'S.No':>4s}  {'Parameter':<24s}{'MSE':>12s}{'MAE':>12s}{'R²':>10s}\n")
                f.write("-" * 60 + "\n")

            for m in per_param_metrics:
                if use_loss_weights == 1:
                    f.write(
                        f"{m['idx']:>4d}  {m['name']:<24s}{m['mse']:>12.6f}{m['mae']:>12.6f}{m['r2']:>10.4f}{m['weight']:>10.2f}\n"
                    )
                else:
                    f.write(
                        f"{m['idx']:>4d}  {m['name']:<24s}{m['mse']:>12.6f}{m['mae']:>12.6f}{m['r2']:>10.4f}\n"
                    )

            # ----- group averages like in your LaTeX table -----
            # index ranges are 1-based in the description
            def _avg_metrics(start_idx, end_idx):
                subset = [m for m in per_param_metrics if start_idx <= m["idx"] <= end_idx]
                if not subset:
                    return (0.0, 0.0, 0.0)
                mse_avg = sum(m["mse"] for m in subset) / len(subset)
                mae_avg = sum(m["mae"] for m in subset) / len(subset)
                r2_avg = sum(m["r2"] for m in subset) / len(subset)
                return (mse_avg, mae_avg, r2_avg)

            f.write("\n")
            # 1–6 and 7–16 like your screenshot
            mse_1_6, mae_1_6, r2_1_6 = _avg_metrics(1, 6)
            mse_7_16, mae_7_16, r2_7_16 = _avg_metrics(7, 16)
            # align with table: 4s + 2 + 24s + 12 + 12 + 10
            f.write(f"{'':>4s}  {'Average of 1 to 6':<24s}{mse_1_6:>12.5f}{mae_1_6:>12.5f}{r2_1_6:>10.3f}\n")
            f.write(f"{'':>4s}  {'Average of 7 to 16':<24s}{mse_7_16:>12.5f}{mae_7_16:>12.5f}{r2_7_16:>10.3f}\n")

            # extra group averages you asked for
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
            # mark training done time
            if hasattr(self, "timing_info"):
                self.timing_info["train_done"] = time.time()

            f.write("================================================================================\n")
            f.write("TIMING SUMMARY\n")
            f.write("================================================================================\n")

            ti = getattr(self, "timing_info", None)
            if ti is not None:
                start_all = ti.get("start_all", None)
                base_done = ti.get("base_done", None)
                temporal_done = ti.get("temporal_done", None)
                train_start = ti.get("train_start", None)
                train_done = ti.get("train_done", None)

                def _fmt_hm(seconds):
                    if seconds is None:
                        return "N/A"
                    mins = int(round(seconds / 60.0))
                    hrs = mins // 60
                    mm = mins % 60
                    return f"{hrs:02d}:{mm:02d} (h:mm)"

                # 1. base
                if start_all and base_done:
                    f.write(f"1. Base train dataset processed: {_fmt_hm(base_done - start_all)}\n")
                else:
                    f.write("1. Base train dataset processed: N/A\n")

                # 2. temporal
                if temporal_done and base_done:
                    f.write(f"2. Temporal dataset processed:   {_fmt_hm(temporal_done - base_done)}\n")
                else:
                    f.write("2. Temporal dataset processed:   N/A (disabled or missing)\n")

                
                # 3. model training
                if train_start and train_done:
                    f.write(f"4. Model training (NN):          {_fmt_hm(train_done - train_start)}\n")
                else:
                    f.write("4. Model training (NN):          N/A\n")

                # 5 and 6 will be added by save_models()
            else:
                f.write("Timing info not available.\n")

        print(f"\n📄 Evaluation report saved to: {report_path}")

    

    def classify_audio(self, audio_path):
        """Classify audio file and predict knob settings"""
        
        # Extract only what is needed (REG and/or CLS). Disabled features are NOT extracted.
        need_clap = (self.feature_usage.get("reg_use_clap", 1) == 1) or (self.feature_usage.get("cls_use_clap", 1) == 1)
        need_mel  = (self.feature_usage.get("reg_use_mel", 1) == 1)  or (self.feature_usage.get("cls_use_mel", 1) == 1)

        clap_feat = None
        mel_feat = None

        if need_clap:
            clap_feat = self.extract_audio_features(audio_path)

        if need_mel:
            mel_feat = self.extract_mel_cnn_features(audio_path)

        # Build classification feature vector
        if self.feature_usage.get("cls_use_clap", 1) == 1 and clap_feat is None:
            raise ValueError("cls_use_clap=1 but CLAP was not extracted.")
        if self.feature_usage.get("cls_use_mel", 1) == 1 and mel_feat is None:
            raise ValueError("cls_use_mel=1 but MEL was not extracted.")

        cls_vec = self._fuse_np(
            clap_feat if clap_feat is not None else np.zeros((self.clap_dim,), dtype=np.float32),
            mel_feat if mel_feat is not None else np.zeros((self.mel_out_dim,), dtype=np.float32),
            task="cls"
        )
        cls_features = cls_vec.reshape(1, -1)

        # Build regression feature vector (for knob prediction)
        if self.feature_usage.get("reg_use_clap", 1) == 1 and clap_feat is None:
            raise ValueError("reg_use_clap=1 but CLAP was not extracted.")
        if self.feature_usage.get("reg_use_mel", 1) == 1 and mel_feat is None:
            raise ValueError("reg_use_mel=1 but MEL was not extracted.")

        reg_vec = self._fuse_np(
            clap_feat if clap_feat is not None else np.zeros((self.clap_dim,), dtype=np.float32),
            mel_feat if mel_feat is not None else np.zeros((self.mel_out_dim,), dtype=np.float32),
            task="reg"
        )

        # Use base_feat as the canonical regression feature vector for the rest of the inference logic
        base_feat = reg_vec

        # ---- Sanity checks (printed once per process) -----------------------
        if not hasattr(self, "_did_print_infer_info"):
            self._did_print_infer_info = True
            print(f"[INF] infer base_feat dim = {base_feat.shape[0]}, "
                  f"trained feature_dim = {getattr(self, 'feature_dim', 'None')}")
            if getattr(self, "_mel_loaded", False):
                print("[INF] mel_crnn weights: LOADED")
            else:
                print("[INF] mel_crnn weights: NOT LOADED (random init)")
        # --------------------------------------------------------------------

        # Ensure inference feature size matches what the loaded regressor expects.
        # With DI removed, a mismatch almost always means train/test settings (feature_usage)
        # are inconsistent, or the wrong model files were loaded.
        target_dim = getattr(self, "feature_dim", base_feat.shape[0])
        if base_feat.shape[0] != target_dim:
            raise ValueError(
                f"[CONFIG ERROR] Regression feature dim mismatch: "
                f"got {base_feat.shape[0]} but model expects {target_dim}. "
                f"Check that train/test use the SAME feature_usage toggles and model artifacts."
            )

        audio_features = base_feat.reshape(1, -1)


        # Predict tone type (RF or NN-head) using CLASSIFICATION features
        if self.classifier_type == "rf":
            tone_pred = self.tone_classifier.predict(cls_features)[0]

            # Robust decoding: normally tone_pred is an encoded int (your current training).
            # If an older artifact was trained on string labels, inverse_transform would fail.
            try:
                tone_name = self.label_encoder.inverse_transform([tone_pred])[0]
            except Exception:
                tone_name = str(tone_pred)

            tone_prob = self.tone_classifier.predict_proba(cls_features)[0]
            confidence = float(np.max(tone_prob))



        elif self.classifier_type == "nn_head":
            if self.tone_classifier_nn is None:
                raise ValueError("classifier_type='nn_head' but tone_classifier_nn is not loaded.")
            x = torch.from_numpy(cls_features).float().to(self.device)
            with torch.no_grad():
                logits = self.tone_classifier_nn(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                tone_pred = int(np.argmax(probs))
            tone_name = self.label_encoder.inverse_transform([tone_pred])[0]
            confidence = float(np.max(probs))

        else:
            raise ValueError(f"Unknown classifier_type='{self.classifier_type}'. Use 'rf' or 'nn_head'.")


        # Predict knob settings using neural network
        knob_settings = {}
        if self.knob_regressor_net is not None:
            # --- CHANGED FOR E2E: DO NOT USE SCALER TRANSFORM ---
            # The E2E training loop fed raw features directly to the network 
            # (relying on BatchNorm). We must do the same here.
            audio_tensor = torch.FloatTensor(audio_features).to(self.device)
            
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
        """Match text description to best tone type and provide knob settings"""
        # Extract text embedding
        text_embed = self.extract_text_features([text_description])[0]
        
        # Calculate similarity with each tone type
        similarities = {}
        for tone_type, tone_embed in self.tone_embeddings.items():
            # Cosine similarity
            similarity = np.dot(text_embed, tone_embed) / (
                np.linalg.norm(text_embed) * np.linalg.norm(tone_embed)
            )
            similarities[tone_type] = similarity
        
        # Get best matching tone
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
        For each index_{name}.csv, generates:
          1. index_{name}_prediction_crnn.csv (n_ params replaced by model preds)
          2. index_{name}_error_crnn.csv (n_ params replaced by absolute error)
        
        Saves these files to the specified save_dir.
        """
        print("\n[INFO] Generating detailed prediction and error CSV reports...")
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Identify datasets to process (Train + any enabled toggles)
        dataset_map = self._discover_datasets()
        datasets_to_process = []
        
        # Always process 'train' if present
        if "train" in dataset_map:
            datasets_to_process.append(("train", dataset_map["train"]))
            
        for ds_key, ds_path in dataset_map.items():
            if ds_key == "train": continue
            # Check if enabled
            toggle_name = f"{ds_key}_aware"
            if self.dataset_toggles.get(toggle_name, 0) == 1:
                datasets_to_process.append((ds_key, ds_path))
        
        # Ensure models are in eval mode
        if self.knob_regressor_net:
            self.knob_regressor_net.eval()
        if self.mel_encoder:
            self.mel_encoder.eval()
        
        for key, csv_path in datasets_to_process:
            print(f"  -> Processing {key} ({os.path.basename(csv_path)})...")
            
            try:
                df = pd.read_csv(csv_path)
                
                # --- FIX: APPLY PREPROCESSING FOR FAIR COMPARISON ---
                # Uses Central Config defaults automatically
                df = self._preprocess_dataset(df, dataset_name=key)
                # ----------------------------------------------------
                
                preds_list = []                
                # Iterate rows to handle file paths and inference correctly (E2E)
                for idx, row in df.iterrows():
                    audio_path, _ = self._resolve_sample_paths(key, row)     # <--- FIXED (Only 2 variables)
                    
                    if audio_path and os.path.exists(audio_path):
                        try:
                            # Replicate classify_audio logic (Respecting Toggles):
                            
                            # 1. Extract only enabled features
                            clap_feat = None
                            if self.feature_usage.get("reg_use_clap", 1) == 1:
                                clap_feat = self.extract_audio_features(audio_path)
                            
                            mel_feat = None
                            if self.feature_usage.get("reg_use_mel", 1) == 1:
                                mel_feat = self.extract_mel_cnn_features(audio_path)

                            # 2. Fuse using the standard helper (handles dim checks automatically)
                            # We pass zeros if None, but _fuse_np will only use them if the toggle is ON.
                            # Since we guarded extraction with the same toggle, this is safe.
                            base_feat = self._fuse_np(
                                clap_feat if clap_feat is not None else np.zeros(self.clap_dim, dtype=np.float32),
                                mel_feat if mel_feat is not None else np.zeros(self.mel_out_dim, dtype=np.float32),
                                task="reg"
                            )
                                    
                            # 3. Inference
                            audio_features = base_feat.reshape(1, -1)                            
                            audio_tensor = torch.FloatTensor(audio_features).to(self.device)
                            
                            with torch.no_grad():
                                knob_predictions = self.knob_regressor_net(audio_tensor).cpu().numpy()[0]
                                preds_list.append(knob_predictions)
                                
                        except Exception as e:
                            print(f"    [WARN] Error predicting row {idx}: {e}")
                            preds_list.append(np.zeros(len(self.knob_params)) + 0.5)
                    else:
                        # Fallback for missing files
                        preds_list.append(np.zeros(len(self.knob_params)) + 0.5)
                
                if not preds_list:
                    print(f"  [WARN] No rows processed for {key}.")
                    continue
                    
                preds_np = np.array(preds_list) # (N_samples, 17)
                
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
                
                path_pred = os.path.join(save_dir, f"{base_name}_prediction_crnn.csv")
                path_err = os.path.join(save_dir, f"{base_name}_error_crnn.csv")
                
                df_pred.to_csv(path_pred, index=False, float_format='%.5f')
                df_error.to_csv(path_err, index=False, float_format='%.5f')
                
                print(f"     Saved: {os.path.basename(path_pred)}")
                print(f"     Saved: {os.path.basename(path_err)}")
                
            except Exception as e:
                print(f"  [ERR] Failed to generate reports for {key}: {e}")

    def save_models(self, save_dir="models"):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)
    
        # Save classifier type + feature usage (simple pkls; no complex metadata system)
        with open(os.path.join(save_dir, 'classifier_type.pkl'), 'wb') as f:
            pickle.dump(self.classifier_type, f)

        with open(os.path.join(save_dir, 'feature_usage.pkl'), 'wb') as f:
            pickle.dump(self.feature_usage, f)

        # Save tone classifier (RF only). For nn_head, store None here and save torch weights separately.
        with open(os.path.join(save_dir, 'tone_classifier.pkl'), 'wb') as f:
            pickle.dump(self.tone_classifier, f)

        if self.classifier_type == "nn_head" and self.tone_classifier_nn is not None:
            torch.save(self.tone_classifier_nn.state_dict(), os.path.join(save_dir, 'tone_classifier_nn.pth'))

            
        # Save label encoder
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
        # Save neural network state dict (if available)
        if self.knob_regressor_net is not None:
            torch.save(
                self.knob_regressor_net.state_dict(),
                os.path.join(save_dir, 'knob_regressor_net.pth')
            )
    
        # Save feature scaler
        with open(os.path.join(save_dir, 'knob_scaler.pkl'), 'wb') as f:
            pickle.dump(self.knob_scaler, f)
    
        # Save feature dimension (use getattr to avoid crash if not set)
        feat_dim = getattr(self, "feature_dim", None)
        with open(os.path.join(save_dir, 'feature_dim.pkl'), 'wb') as f:
            pickle.dump(feat_dim, f)

        # NEW: Save mel CRNN weights so inference uses the exact same encoder
        uses_mel = (self.feature_usage.get("reg_use_mel", 1) == 1) or (self.feature_usage.get("cls_use_mel", 1) == 1)
        if uses_mel and hasattr(self, "mel_encoder") and self.mel_encoder is not None:
            torch.save(self.mel_encoder.state_dict(), os.path.join(save_dir, 'mel_crnn.pth'))


        # Save tone embeddings for text-based tone matching
        with open(os.path.join(save_dir, 'tone_embeddings.pkl'), 'wb') as f:
            pickle.dump(self.tone_embeddings, f)
        
        # --- CHANGE START ---
        
        # Generate the requested CSV reports for all processed datasets
        self.generate_detailed_csv_reports(save_dir=save_dir)
        # --- CHANGE END ---
        
        save_end = time.time()

        # Print what artifacts were actually saved this run
        try:
            saved_files = [
                "classifier_type.pkl",
                "feature_usage.pkl",
                "tone_classifier.pkl",
                "label_encoder.pkl",
                "knob_scaler.pkl",
                "feature_dim.pkl",
                "tone_embeddings.pkl",
            ]

            if self.knob_regressor_net is not None:
                saved_files.append("knob_regressor_net.pth")

            uses_mel = (self.feature_usage.get("reg_use_mel", 1) == 1) or (self.feature_usage.get("cls_use_mel", 1) == 1)
            if uses_mel:
                saved_files.append("mel_crnn.pth")

            if self.classifier_type == "nn_head" and self.tone_classifier_nn is not None:
                saved_files.append("tone_classifier_nn.pth")

            # Only show those that exist on disk
            existing = [f for f in saved_files if os.path.exists(os.path.join(save_dir, f))]
            missing  = [f for f in saved_files if f not in existing]

            print("[SAVE] Artifacts written:")
            for f in existing:
                print(f"  - {f}")
            if missing:
                print("[SAVE] Expected-but-not-present (usually OK if feature/model disabled):")
                for f in missing:
                    print(f"  - {f}")

        except Exception as e:
            print(f"[WARN] Could not list saved artifacts: {e}")

        print(f"✅ Models saved successfully to {save_dir}/")

        # record in timing info        
        
        if not hasattr(self, "timing_info"):
            self.timing_info = {}
        self.timing_info["models_saved"] = save_end

        # if we know when we started, we can compute total
        if "start_all" in self.timing_info:
            self.timing_info["all_done"] = save_end

        # append to last report if we have one
        report_path = getattr(self, "last_report_path", None)
        if report_path and os.path.exists(report_path):
            try:
                with open(report_path, "a", encoding="utf-8") as f:
                    f.write("\n")
                    f.write("---- Save/Final timings -------------------------------------------------\n")

                    def _fmt_hm(seconds):
                        if seconds is None:
                            return "N/A"
                        mins = int(round(seconds / 60.0))
                        hrs = mins // 60
                        mm = mins % 60
                        return f"{hrs:02d}:{mm:02d} (h:mm)"

                    models_saved = self.timing_info.get("models_saved", None)
                    start_all = self.timing_info.get("start_all", None)
                    f.write(
                        "5. Models saved:                 "
                        + (_fmt_hm(models_saved - start_all) if (models_saved and start_all) else "N/A")
                        + "\n"
                    )

                    all_done = self.timing_info.get("all_done", None)
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

        # Load classifier type + feature usage if present (keeps train/test aligned without a complex system)
        ctype_path = os.path.join(save_dir, 'classifier_type.pkl')
        if os.path.exists(ctype_path):
            with open(ctype_path, 'rb') as f:
                self.classifier_type = pickle.load(f)

        fusage_path = os.path.join(save_dir, 'feature_usage.pkl')
        if os.path.exists(fusage_path):
            with open(fusage_path, 'rb') as f:
                self.feature_usage = pickle.load(f)

        with open(os.path.join(save_dir, 'tone_classifier.pkl'), 'rb') as f:
            self.tone_classifier = pickle.load(f)

        # If RF is requested, tone_classifier.pkl must contain a valid sklearn model
        if str(getattr(self, "classifier_type", "rf")).lower() in ("rf", "randomforest") and self.tone_classifier is None:
            raise RuntimeError("classifier_type='rf' but tone_classifier.pkl contains None. Re-train or check save_models().")

        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        
        # NEW: load feature_dim if available
        feature_dim_path = os.path.join(save_dir, 'feature_dim.pkl')
        if os.path.exists(feature_dim_path):
            with open(feature_dim_path, 'rb') as f:
                self.feature_dim = pickle.load(f)
        
        else:
            # backward compat: old models were 512-D
            self.feature_dim = 512

        # Recompute dims from toggles (ensures inference vectors match)
        self.feature_dim = self._get_feature_dim_for_task("reg")
        self.cls_feature_dim = self._get_feature_dim_for_task("cls")

        # Load mel CRNN weights only if MEL is used by either regression or classification
        mel_path = os.path.join(save_dir, 'mel_crnn.pth')

        self._mel_loaded = False
        uses_mel = (self.feature_usage.get("reg_use_mel", 1) == 1) or (self.feature_usage.get("cls_use_mel", 1) == 1)

        if uses_mel:
            if os.path.exists(mel_path):
                state = torch.load(mel_path, map_location=self.device)
                self.mel_encoder.load_state_dict(state)
                self.mel_encoder.to(self.device)
                self.mel_encoder.eval()
                self._mel_loaded = True
            else:
                print("[WARN] mel_crnn.pth not found but MEL is enabled in feature_usage. Expect degraded performance.")
        else:
            # MEL disabled for both reg/cls → no need to load mel encoder weights
            self._mel_loaded = False


        # Load NN-head classifier if requested
        self.tone_classifier_nn = None
        if getattr(self, "classifier_type", "rf") == "nn_head":
            nn_path = os.path.join(save_dir, 'tone_classifier_nn.pth')
            if os.path.exists(nn_path):
                num_classes = len(self.label_encoder.classes_)
                # cls_feature_dim depends on toggles; compute it
                self.cls_feature_dim = self._get_feature_dim_for_task("cls")
                self.tone_classifier_nn = ToneClassifierHead(input_dim=self.cls_feature_dim, num_classes=num_classes).to(self.device)
                self.tone_classifier_nn.load_state_dict(torch.load(nn_path, map_location=self.device))
                self.tone_classifier_nn.eval()
            else:
                print("[WARN] tone_classifier_nn.pth not found but classifier_type='nn_head'.")

        # Load neural network
        net_path = os.path.join(save_dir, 'knob_regressor_net.pth')


        if os.path.exists(net_path):
            self.knob_regressor_net = KnobParameterNet(
                input_dim=self.feature_dim,
                hidden_dims=[256, 128, 64],
                output_dim=len(self.knob_params)
            ).to(self.device)
            self.knob_regressor_net.load_state_dict(torch.load(net_path, map_location=self.device))
            self.knob_regressor_net.eval()

                    
        # Load scaler
        with open(os.path.join(save_dir, 'knob_scaler.pkl'), 'rb') as f:
            self.knob_scaler = pickle.load(f)
        
        with open(os.path.join(save_dir, 'tone_embeddings.pkl'), 'rb') as f:
            self.tone_embeddings = pickle.load(f)
        
        # NOTE: Do NOT load dataset here. Inference must work without index CSVs.
        # Training code will call load_data() explicitly.
        pass


        # Print what will be used for inference/testing
        try:
            print("[LOAD] Inference configuration:")
            print(f"  - classifier_type      = {getattr(self, 'classifier_type', 'rf')}")
            print(f"  - feature_usage        = {getattr(self, 'feature_usage', {})}")
            print(f"  - reg feature_dim      = {getattr(self, 'feature_dim', None)}")
            print(f"  - cls feature_dim      = {getattr(self, 'cls_feature_dim', None)}")
            print(f"  - mel_crnn weights     = {'LOADED' if getattr(self, '_mel_loaded', False) else 'NOT LOADED'}")
            print(f"  - knob_regressor_net   = {'LOADED' if (self.knob_regressor_net is not None) else 'NOT LOADED'}")
            if getattr(self, "classifier_type", "rf") == "rf":
                print(f"  - tone_classifier (RF) = {'LOADED' if (self.tone_classifier is not None) else 'NOT LOADED'}")
            elif getattr(self, "classifier_type", "rf") == "nn_head":
                print(f"  - tone_classifier_nn   = {'LOADED' if (self.tone_classifier_nn is not None) else 'NOT LOADED'}")
        except Exception as e:
            print(f"[WARN] Could not print load summary: {e}")

        print(f"Models loaded from {save_dir}/")


def main():

    # --- AUTO-CLEANUP START ---
    import shutil
    print("\n[PRE-FLIGHT CLEANUP] Removing old cache and processed indices...")
    
    # 1. Clean Cache Folder
    # Adjust "Data" if your folder is named differently relative to this script
    cache_dir = os.path.join("Data", "cached_features_crnn")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"   -> Deleted old cache folder: {cache_dir}")
        except Exception as e:
            print(f"   [WARN] Failed to delete cache: {e}")
    else:
        print(f"   -> Cache folder not found (Clean).")

    # 2. Clean Processed CSVs (index_*_processed.csv)
    index_dir = os.path.join("Data", "indexes")
    if os.path.exists(index_dir):
        for f in os.listdir(index_dir):
            if f.endswith("_processed.csv"):
                try:
                    os.remove(os.path.join(index_dir, f))
                    print(f"   -> Deleted old processed index: {f}")
                except Exception as e:
                    print(f"   [WARN] Failed to delete {f}: {e}")
    print("[PRE-FLIGHT CLEANUP] Done.\n")
    # --- AUTO-CLEANUP END ---
    
    # simple training entrypoint using the new auto-discovery
    classifier = MusicToneClassifier(data_dir="Data")
    classifier.train_models(
        max_samples_per_class=20,   # or your usual value
        epochs=300,
        batch_size=32,
        learning_rate=0.001
    )
    # optional: if you still want to save
    classifier.save_models()
    print("\nTraining completed! Models saved.")

if __name__ == "__main__":
    main()
