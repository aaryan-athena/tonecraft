import json
import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from music_tone_classifier import MusicToneClassifier
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from demucs.pretrained import get_model
from demucs.apply import apply_model
from io import BytesIO
import zipfile

# --- Load Global Ranges ---
def load_global_ranges():
    """Load parameter ranges from global ranges.json"""
    ranges_path = os.path.join(os.path.dirname(__file__), "global ranges.json")
    with open(ranges_path, 'r') as f:
        data = json.load(f)
    return data.get("Global", {})

# --- Initialize AI Model ---
print("Loading AI models...")
classifier = MusicToneClassifier()
classifier.load_models(save_dir="models")
global_ranges = load_global_ranges()
print("AI models loaded successfully!")

# --- Denormalization Function ---
def denormalize_value(normalized_value, param_name):
    """Convert normalized value [0, 1] to actual parameter range"""
    if param_name not in global_ranges:
        return normalized_value
    
    min_val, max_val = global_ranges[param_name]
    return min_val + normalized_value * (max_val - min_val)

# --- AI Model Integration ---
def run_ai_model(input_data: dict) -> dict:
    """
    Run the trained AI model to predict guitar tone parameters.
    Accepts either 'text_input' (text description) or 'audio_path' (audio file path).
    Returns denormalized parameter values ready for the plugin.
    """
    try:
        if 'text_input' in input_data:
            text_input = input_data['text_input']
            print(f"AI Model: Processing text prompt -> '{text_input}'")
            
            # Use text-to-tone matching
            result = classifier.match_text_to_tone(text_input)
            knob_settings = result['knob_settings']
            tone_type = result['tone_type']
            confidence = result['confidence']
            
            print(f"AI Model: Matched to '{tone_type}' tone (confidence: {confidence:.2%})")
            
        elif 'audio_path' in input_data:
            audio_path = input_data['audio_path']
            print(f"AI Model: Processing audio file -> '{audio_path}'")
            
            # Validate audio file exists
            if not os.path.exists(audio_path):
                print(f"AI Model: Error - Audio file not found: {audio_path}")
                return {}
            
            # Use audio classification
            result = classifier.classify_audio(audio_path)
            knob_settings = result['knob_settings']
            tone_type = result['tone_type']
            confidence = result['confidence']
            
            print(f"AI Model: Classified as '{tone_type}' tone (confidence: {confidence:.2%})")
            
        else:
            print("AI Model: Error - Invalid request (missing 'text_input' or 'audio_path')")
            return {}
        
        # Denormalize all parameters to their actual ranges
        parameters = []
        for param_name, normalized_value in knob_settings.items():
            denormalized_value = denormalize_value(normalized_value, param_name)
            parameters.append({
                "name": param_name,
                "value": round(denormalized_value, 4)
            })
        
        print(f"AI Model: Generated {len(parameters)} parameters")
        
        # Return in the expected format with tone type and confidence
        # Convert numpy types to native Python types for JSON serialization
        return {
            "PARAMETERS": parameters,
            "tone_type": str(tone_type),
            "confidence": float(confidence)
        }
        
    except Exception as e:
        print(f"AI Model: Error during prediction - {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# --- Flask Server Setup ---
app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

@app.route('/')
def index():
    """Serve the web interface"""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    This is the main endpoint for the AI server.
    It expects a JSON payload with either a 'text_input' or 'audio_path' key.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    content = request.get_json()
    
    # Run the AI model with the received content
    ai_result = run_ai_model(content)

    if not ai_result:
        return jsonify({"error": "Invalid input to AI model"}), 400
        
    # Return the AI's parameter predictions as a JSON response
    return jsonify(ai_result)

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    """
    Endpoint for audio file uploads.
    Accepts multipart/form-data with an 'audio' file.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Create uploads directory if it doesn't exist
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Save the uploaded file temporarily
    import time
    timestamp = int(time.time() * 1000)
    filename = f"audio_{timestamp}_{audio_file.filename}"
    filepath = os.path.join(uploads_dir, filename)
    audio_file.save(filepath)
    
    try:
        # Run the AI model with the audio file path
        ai_result = run_ai_model({'audio_path': filepath})
        
        if not ai_result:
            return jsonify({"error": "Failed to process audio file"}), 400
        
        return jsonify(ai_result)
    
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {filepath}: {e}")

# --- Splitter (StemFlow) Functions ---
demucs_model = None
demucs_device = None

def load_demucs_model():
    """Load the Demucs model for stem separation"""
    global demucs_model, demucs_device
    if demucs_model is None:
        try:
            print("Loading Demucs model...")
            model_name = 'htdemucs'  # Valid model name for 4-stem configuration
            demucs_model = get_model(model_name)
            demucs_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            demucs_model.to(demucs_device)
            print(f"Demucs model loaded successfully on {demucs_device}!")
        except Exception as e:
            print(f"Error loading Demucs model: {e}")
            raise ValueError(f"Error loading model: {e}")
    return demucs_model, demucs_device

def convert_to_wav(input_path):
    """Convert audio to WAV format if necessary"""
    try:
        audio = AudioSegment.from_file(input_path)
        temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        audio.export(temp_wav_path, format="wav")
        return temp_wav_path
    except Exception as e:
        raise ValueError(f"Audio conversion failed: {e}")

def load_audio_for_demucs(file_path, sample_rate=44100):
    """Load audio file for Demucs processing"""
    try:
        wav, sr = sf.read(file_path, dtype='float32')
        wav = torch.from_numpy(wav.T if wav.ndim > 1 else wav[np.newaxis, :])
        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
        return wav, sample_rate
    except Exception as e:
        raise ValueError(f"Error loading audio: {e}")

def separate_stems(model, wav, device):
    """Perform stem separation using Demucs"""
    wav = wav.unsqueeze(0).to(device)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        stems = apply_model(model, wav)
    return stems

def save_stems_as_bytes(stems, sample_rate, stem_names):
    """Save separated stems as bytes"""
    stem_files = {}
    # stems shape: (batch, num_sources, channels, samples)
    # When iterating, idx is batch index, stem is (num_sources, channels, samples)
    for idx, stem in enumerate(stems):
        try:
            if stem.ndimension() == 3:  # (num_sources, channels, samples)
                # b iterates over the sources: drums, bass, other, vocals
                for b in range(stem.shape[0]):
                    stem_b = stem[b].cpu().numpy()  # (channels, samples)
                    buffer = BytesIO()
                    # Transpose to (samples, channels) for soundfile
                    sf.write(buffer, stem_b.T, sample_rate, format='WAV')
                    buffer.seek(0)
                    # Use b (source index) not idx (batch index) for correct stem name
                    stem_files[stem_names[b]] = buffer
            else:
                raise ValueError(f"Unexpected tensor shape: {stem.shape}")
        except Exception as e:
            print(f"Error saving stem {b}: {e}")
            continue
    return stem_files

@app.route('/split', methods=['POST'])
def split_audio():
    """
    Endpoint for audio stem separation.
    Accepts multipart/form-data with an 'audio' file.
    Returns a ZIP file containing all separated stems.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Create uploads directory if it doesn't exist
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Save the uploaded file temporarily
    import time
    timestamp = int(time.time() * 1000)
    original_filename = os.path.splitext(audio_file.filename)[0]
    filename = f"split_{timestamp}_{audio_file.filename}"
    filepath = os.path.join(uploads_dir, filename)
    audio_file.save(filepath)
    
    wav_file_path = None
    
    try:
        # Load Demucs model
        model, device = load_demucs_model()
        
        # Convert input file to WAV if necessary
        wav_file_path = convert_to_wav(filepath)
        
        # Load audio into torch format
        wav, sample_rate = load_audio_for_demucs(wav_file_path)
        
        # Perform stem separation
        stems = separate_stems(model, wav, device)
        
        # Stem names for the 4-stem configuration
        stem_names = ["drums", "bass", "other", "vocals"]
        
        # Save stems as bytes
        stem_files = save_stems_as_bytes(stems, sample_rate, stem_names)
        
        # Create a ZIP file containing all stems
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for stem_name, stem_buffer in stem_files.items():
                stem_buffer.seek(0)
                zip_file.writestr(f"{original_filename}_{stem_name}.wav", stem_buffer.read())
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{original_filename}_stems.zip"
        )
        
    except Exception as e:
        print(f"Error during stem separation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
    
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            if wav_file_path and os.path.exists(wav_file_path):
                os.remove(wav_file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary files: {e}")

@app.route('/split_individual', methods=['POST'])
def split_audio_individual():
    """
    Endpoint for audio stem separation that returns individual stems info.
    Use this for previewing stems before downloading.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Create uploads directory if it doesn't exist
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Save the uploaded file temporarily
    import time
    timestamp = int(time.time() * 1000)
    original_filename = os.path.splitext(audio_file.filename)[0]
    filename = f"split_{timestamp}_{audio_file.filename}"
    filepath = os.path.join(uploads_dir, filename)
    audio_file.save(filepath)
    
    wav_file_path = None
    
    try:
        # Load Demucs model
        model, device = load_demucs_model()
        
        # Convert input file to WAV if necessary
        wav_file_path = convert_to_wav(filepath)
        
        # Load audio into torch format
        wav, sample_rate = load_audio_for_demucs(wav_file_path)
        
        # Perform stem separation
        stems = separate_stems(model, wav, device)
        
        # Stem names for the 4-stem configuration
        stem_names = ["drums", "bass", "other", "vocals"]
        
        # Save stems as bytes
        stem_files = save_stems_as_bytes(stems, sample_rate, stem_names)
        
        # Save stems to temporary files and return their paths
        stem_info = []
        for stem_name, stem_buffer in stem_files.items():
            stem_filename = f"{original_filename}_{stem_name}_{timestamp}.wav"
            stem_filepath = os.path.join(uploads_dir, stem_filename)
            with open(stem_filepath, 'wb') as f:
                f.write(stem_buffer.read())
            stem_info.append({
                "name": stem_name,
                "filename": stem_filename,
                "download_url": f"/download_stem/{stem_filename}"
            })
        
        return jsonify({
            "success": True,
            "original_filename": original_filename,
            "stems": stem_info
        })
        
    except Exception as e:
        print(f"Error during stem separation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
    
    finally:
        # Clean up original input files (keep stem files for download)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            if wav_file_path and os.path.exists(wav_file_path):
                os.remove(wav_file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary files: {e}")

@app.route('/download_stem/<filename>')
def download_stem(filename):
    """Download an individual stem file"""
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    filepath = os.path.join(uploads_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(
        filepath,
        mimetype='audio/wav',
        as_attachment=True,
        download_name=filename
    )

# --- Main Server Logic ---
def main():
    print("Starting AI Tone Crafting Server (Flask)...")
    # Get port from environment variable (required for Render deployment)
    # Default to 5000 for local development
    port = int(os.environ.get('PORT', 5000))
    # Use 0.0.0.0 to make server accessible externally (required for Render)
    # Use 127.0.0.1 for local development only
    host = '0.0.0.0' if os.environ.get('PORT') else '127.0.0.1'
    print(f"Server running on {host}:{port}")
    app.run(host=host, port=port)

if __name__ == "__main__":
    main()