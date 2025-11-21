#!/usr/bin/env python3
"""
Example usage of the AI Flask Server API
Shows how to make requests from Python code
"""

import requests
import json

def predict_from_text(text_description):
    """Get parameter predictions from text description"""
    url = "http://127.0.0.1:5000/predict"
    payload = {"text_input": text_description}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def predict_from_audio(audio_path):
    """Get parameter predictions from audio file"""
    url = "http://127.0.0.1:5000/predict"
    payload = {"audio_path": audio_path}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def print_parameters(result):
    """Pretty print the parameter results"""
    if not result or 'PARAMETERS' not in result:
        print("No parameters received")
        return
    
    print("\nPredicted Parameters:")
    print("-" * 50)
    for param in result['PARAMETERS']:
        print(f"{param['name']:20} = {param['value']:8.4f}")
    print("-" * 50)

def main():
    print("🎸 AI Flask Server - Example Usage\n")
    
    # Example 1: Text-based prediction
    print("Example 1: Text-based prediction")
    print("=" * 50)
    
    text_prompts = [
        "heavy metal with lots of distortion",
        "clean jazz guitar tone",
        "ambient spacey reverb"
    ]
    
    for prompt in text_prompts:
        print(f"\nPrompt: '{prompt}'")
        result = predict_from_text(prompt)
        if result:
            print_parameters(result)
    
    # Example 2: Audio-based prediction
    print("\n\nExample 2: Audio-based prediction")
    print("=" * 50)
    
    # Replace with actual audio file path
    audio_file = "Data/generated_audio/train/sample.wav"
    print(f"\nAudio file: {audio_file}")
    
    result = predict_from_audio(audio_file)
    if result:
        print_parameters(result)
    else:
        print("(Audio file not found - this is just an example)")
    
    # Example 3: Using the results in your plugin
    print("\n\nExample 3: Using results in your plugin")
    print("=" * 50)
    
    result = predict_from_text("blues crunch tone")
    if result and 'PARAMETERS' in result:
        print("\nIn your plugin, you can directly use these values:")
        print("```cpp")
        for param in result['PARAMETERS'][:3]:  # Show first 3 as example
            param_name = param['name']
            param_value = param['value']
            print(f"setParameter(\"{param_name}\", {param_value});")
        print("// ... and so on for all 17 parameters")
        print("```")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to server!")
        print("Make sure the server is running:")
        print("  python API/ai_flask_server.py")
