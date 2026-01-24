#!/usr/bin/env python3
"""
Tone Classifier Wrapper - Unified Interface for Audio and Text Models
Routes audio classification to guitar_audio model and text-to-tone to guitar_text model.
"""

import os
import sys
import importlib.util

# Get the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_module_from_path(module_name, file_path):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load the specialized classifier modules dynamically to avoid naming conflicts
audio_classifier_path = os.path.join(current_dir, 'guitar_audio', 'music_tone_classifier.py')
text_classifier_path = os.path.join(current_dir, 'guitar_text', 'music_tone_classifier.py')

# These will be loaded lazily in ToneCraftClassifier.__init__
_audio_module = None
_text_module = None


class ToneCraftClassifier:
    """
    Unified classifier that routes requests to specialized models:
    - Audio files -> guitar_audio model (CRNN + Mel spectrogram based)
    - Text prompts -> guitar_text model (CLAP text embedding based)
    """
    
    def __init__(self):
        """Initialize both specialized classifiers"""
        global _audio_module, _text_module
        
        print("=" * 60)
        print("Initializing ToneCraft Dual-Model System")
        print("=" * 60)
        
        # Load audio classifier module (for classify_audio)
        print("\n[1/2] Loading Audio Classification Model (CRNN)...")
        if _audio_module is None:
            _audio_module = load_module_from_path('guitar_audio_classifier', audio_classifier_path)
        AudioClassifier = _audio_module.MusicToneClassifier
        self.audio_classifier = AudioClassifier()
        
        # Load text classifier module (for match_text_to_tone)
        print("\n[2/2] Loading Text-to-Tone Model (MLP)...")
        if _text_module is None:
            _text_module = load_module_from_path('guitar_text_classifier', text_classifier_path)
        TextClassifier = _text_module.MusicToneClassifier
        self.text_classifier = TextClassifier()
        
        print("\n" + "=" * 60)
        print("ToneCraft Models Ready!")
        print("=" * 60)
    
    def load_models(self, audio_model_dir="guitar_audio/models", text_model_dir="guitar_text/models"):
        """
        Load pre-trained models for both classifiers.
        
        Args:
            audio_model_dir: Path to audio model directory
            text_model_dir: Path to text model directory
        """
        print(f"\nLoading audio models from: {audio_model_dir}")
        self.audio_classifier.load_models(save_dir=audio_model_dir)
        
        print(f"\nLoading text models from: {text_model_dir}")
        self.text_classifier.load_models(save_dir=text_model_dir)
        
        print("\nAll models loaded successfully!")
    
    def classify_audio(self, audio_path):
        """
        Classify an audio file and predict knob settings.
        Uses the guitar_audio model (CRNN + Mel spectrogram).
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            dict with 'tone_type', 'confidence', 'knob_settings'
        """
        return self.audio_classifier.classify_audio(audio_path)
    
    def match_text_to_tone(self, text_description):
        """
        Match a text description to knob parameters.
        Uses the guitar_text model (CLAP text embeddings + MLP).
        
        Args:
            text_description: Text prompt describing the desired tone
            
        Returns:
            dict with 'tone_type', 'confidence', 'knob_settings'
        """
        return self.text_classifier.match_text_to_tone(text_description)


# For backward compatibility - if someone imports MusicToneClassifier from this module
MusicToneClassifier = ToneCraftClassifier
