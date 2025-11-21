#!/usr/bin/env python3
"""
Music Tone Classifier - Inference Only
Loads pre-trained models and makes predictions
No training data required
"""

import os
import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import ClapModel, ClapProcessor
import pickle
import warnings
warnings.filterwarnings('ignore')

class KnobParameterNet(nn.Module):
    """Neural Network for predicting all 17 knob parameters"""
    
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64], output_dim=17, dropout_rate=0.3):
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

class MusicToneClassifier:
    """Inference-only classifier for guitar tone prediction"""
    
    def __init__(self):
        # Load CLAP model
        print("Loading CLAP model...")
        self.clap_model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
        self.clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
        
        # Initialize components
        self.label_encoder = None
        self.tone_classifier = None
        self.knob_regressor_net = None
        self.knob_scaler = None
        self.tone_embeddings = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parameter names
        self.knob_params = [
            'distortiondrive', 'overdrivedrive', 'distortiontone', 
            'eqbass', 'eqmid', 'eqtreble', 
            'chorusrate', 'chorusdepth', 'chorusmix', 
            'delaytime', 'delayfeedback', 'delaymix', 
            'reverbt60', 'reverbsize', 'reverbwet', 'reverbdamp', 
            'mastervolume'
        ]
        
        # Average knob settings per tone type (fallback for text-to-tone)
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
    
    def extract_audio_features(self, audio_path):
        """Extract CLAP embeddings from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=48000, duration=10.0)
            
            # Get CLAP audio embedding
            inputs = self.clap_processor(audios=audio, return_tensors="pt", sampling_rate=48000)
            
            with torch.no_grad():
                audio_embed = self.clap_model.get_audio_features(**inputs)
                
            return audio_embed.numpy().flatten()
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros(512)
    
    def extract_text_features(self, text_list):
        """Extract CLAP embeddings from text descriptions"""
        inputs = self.clap_processor(text=text_list, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            text_embed = self.clap_model.get_text_features(**inputs)
            
        return text_embed.numpy()
    
    def classify_audio(self, audio_path):
        """Classify audio file and predict knob settings"""
        # Extract audio features
        audio_features = self.extract_audio_features(audio_path)
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
                knob_settings[param] = float(knob_predictions[i])
        else:
            # Fallback to defaults
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
        
        best_tone = max(similarities, key=similarities.get)
        confidence = similarities[best_tone]
        
        return {
            'tone_type': best_tone,  # For reference only
            'confidence': confidence,
            'knob_settings': knob_settings,
            'all_similarities': similarities
        }
    
    def _match_text_to_tone_fallback(self, text_description):
        """Fallback method using predefined tone defaults"""
        text_embed = self.extract_text_features([text_description])[0]
        
        similarities = {}
        for tone_type, tone_embed in self.tone_embeddings.items():
            similarity = np.dot(text_embed, tone_embed) / (
                np.linalg.norm(text_embed) * np.linalg.norm(tone_embed)
            )
            similarities[tone_type] = similarity
        
        best_tone = max(similarities, key=similarities.get)
        confidence = similarities[best_tone]
        
        knob_settings = self.tone_defaults.get(best_tone, self.tone_defaults['Clean'])
        
        return {
            'tone_type': best_tone,
            'confidence': confidence,
            'knob_settings': knob_settings,
            'all_similarities': similarities
        }
    
    def load_models(self, save_dir="models"):
        """Load pre-trained models"""
        # Load tone classifier
        with open(os.path.join(save_dir, 'tone_classifier.pkl'), 'rb') as f:
            self.tone_classifier = pickle.load(f)
        
        # Load label encoder
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load neural network
        net_path = os.path.join(save_dir, 'knob_regressor_net.pth')
        if os.path.exists(net_path):
            self.knob_regressor_net = KnobParameterNet(
                input_dim=512, 
                hidden_dims=[256, 128, 64], 
                output_dim=len(self.knob_params)
            ).to(self.device)
            self.knob_regressor_net.load_state_dict(torch.load(net_path, map_location=self.device))
            self.knob_regressor_net.eval()
        
        # Load scaler
        with open(os.path.join(save_dir, 'knob_scaler.pkl'), 'rb') as f:
            self.knob_scaler = pickle.load(f)
        
        # Load tone embeddings
        with open(os.path.join(save_dir, 'tone_embeddings.pkl'), 'rb') as f:
            self.tone_embeddings = pickle.load(f)
        
        print(f"Models loaded from {save_dir}/")
