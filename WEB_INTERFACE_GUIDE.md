# ToneCraft AI - Web Interface Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install flask-cors
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python ai_flask_server.py
```

The server will start on `http://127.0.0.1:5000`

### 3. Open the Web Interface

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## Features

### Text Input Mode
- Describe your desired guitar tone in natural language
- Examples: "heavy metal distortion", "clean jazz tone", "bluesy crunch"
- Click the quick example buttons for inspiration
- Click "Generate Tone Parameters" to get AI predictions

### Audio Upload Mode (Coming Soon)
- Upload an audio file to analyze its tone
- Requires additional server-side implementation for file handling

### Results Display
- View all 17 parameter values with visual knob representations
- See percentage values (0-100%) for each parameter
- View actual parameter values in their correct ranges
- Color-coded by effect category:
  - 🔴 Distortion/Overdrive (Red-Orange)
  - 🔵 EQ (Blue-Cyan)
  - 🟣 Reverb (Purple-Pink)
  - 🟢 Delay (Green-Teal)
  - 🟡 Chorus (Yellow-Amber)
  - 🟣 Master (Indigo-Purple)

### Export
- Export all parameters as JSON file
- Ready to use in your plugin or DAW

## Design Features

- Modern, sleek dark theme with gradient backgrounds
- Responsive design (works on desktop, tablet, mobile)
- Smooth animations and transitions
- Tailwind CSS for styling
- Visual knob representations with progress bars
- Color-coded parameter categories

## Troubleshooting

**Server won't start?**
- Make sure flask-cors is installed: `pip install flask-cors`
- Check that all models are in the `models/` directory
- Verify `global ranges.json` exists

**Can't connect to server?**
- Make sure the server is running on port 5000
- Check the browser console for errors
- Ensure no firewall is blocking localhost connections

**No results showing?**
- Check the browser console for errors
- Verify the server is responding at `http://127.0.0.1:5000/predict`
- Make sure you entered a text description

## API Endpoint

The web interface uses the `/predict` endpoint:

```
POST http://127.0.0.1:5000/predict
Content-Type: application/json

{
  "text_input": "heavy metal distortion"
}
```

Response:
```json
{
  "PARAMETERS": [
    {"name": "distortiondrive", "value": 0.8523},
    {"name": "overdrivedrive", "value": 0.3421},
    ...
  ]
}
```

## Next Steps

1. Start the server: `python ai_flask_server.py`
2. Open browser to `http://127.0.0.1:5000`
3. Try different tone descriptions
4. Export parameters for your plugin

Enjoy crafting tones with AI! 🎸
