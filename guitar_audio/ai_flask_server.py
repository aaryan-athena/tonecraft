import json
import os
from flask import Flask, request, jsonify
from music_tone_classifier import MusicToneClassifier

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
        
        # Return in the expected format
        return {"PARAMETERS": parameters}
        
    except Exception as e:
        print(f"AI Model: Error during prediction - {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# --- Flask Server Setup ---
app = Flask(__name__)

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

# --- Main Server Logic ---
def main():
    print("Starting AI Tone Crafting Server (Flask)...")
    # host='127.0.0.1' makes the server only accessible from the local machine
    # port=5000 is the standard port for Flask development
    app.run(host='127.0.0.1', port=5000)

if __name__ == "__main__":
    main()