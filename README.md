# AI Flask Server - Integration Complete ✅

## What Changed

The Flask API server has been updated to use your **trained machine learning models** instead of random values. The server now provides real AI-powered guitar tone parameter predictions.

## Key Features

✅ **Text-to-Tone Matching** - Describe the tone you want in natural language  
✅ **Audio Classification** - Analyze audio files to extract tone parameters  
✅ **Automatic Denormalization** - Returns plugin-ready values in correct ranges  
✅ **High Accuracy** - 91.88% tone classification accuracy  
✅ **17 Parameters** - Complete knob settings for all effects  

## Quick Start

### 1. Make sure models are trained
```bash
python train_models.py
```

### 2. Start the server
```bash
python API/ai_flask_server.py
```

### 3. Test the API
```bash
python test_api.py
```

## How It Works

### Input Processing
- **Text Input**: Uses CLAP embeddings to match text descriptions to tone types
- **Audio Input**: Extracts audio features and classifies the tone

### Model Pipeline
1. **Tone Classification** (Random Forest) → Identifies tone type (Clean, Crunch, High-Gain, Wellness)
2. **Parameter Prediction** (Neural Network) → Predicts normalized knob values [0, 1]
3. **Denormalization** → Converts to actual parameter ranges using `global ranges.json`

### Output Format
Returns JSON with 17 denormalized parameters ready for your plugin:
```json
{
  "PARAMETERS": [
    {"name": "distortiondrive", "value": 0.8523},
    {"name": "eqbass", "value": 2.4567},
    ...
  ]
}
```

## Files

- **ai_flask_server.py** - Main Flask server with AI integration
- **global ranges.json** - Parameter min/max ranges for denormalization
- **API_USAGE.md** - Complete API documentation
- **example_usage.py** - Python code examples
- **test_api.py** - Automated testing script (in root directory)

## Model Performance

Based on test results:
- **Tone Classification Accuracy**: 91.88%
- **Parameter Prediction R²**: 0.4929
- **Mean Absolute Error**: 0.088 (normalized scale)

### Per-Class Performance
| Tone Type | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Clean | 91.9% | 95.8% | 93.8% |
| Crunch | 87.6% | 86.3% | 86.9% |
| High-Gain | 98.4% | 91.1% | 94.6% |
| Wellness | 89.6% | 95.5% | 92.5% |

## Integration with Plugin

Your plugin can now make HTTP POST requests to `http://127.0.0.1:5000/predict` and receive real AI predictions instead of random values.

Example request:
```json
POST http://127.0.0.1:5000/predict
Content-Type: application/json

{
  "text_input": "heavy metal distortion"
}
```

Example response:
```json
{
  "PARAMETERS": [
    {"name": "distortiondrive", "value": 0.8523},
    {"name": "overdrivedrive", "value": 0.3421},
    ...all 17 parameters...
  ]
}
```

## Next Steps

1. ✅ Models trained and saved
2. ✅ Flask server updated with AI integration
3. ✅ Denormalization implemented
4. ✅ Testing scripts created
5. 🔄 Test with your plugin
6. 🔄 Deploy to production (if needed)

## Troubleshooting

**Server won't start?**
- Make sure all models are in `models/` directory
- Check that `global ranges.json` exists in `API/` directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Getting random values?**
- Old server code has been replaced
- Restart the server to load the new code

**Low accuracy?**
- Train with more samples: `python train_advanced.py --samples 500`
- Use all data: `python train_advanced.py --samples -1`

## Support

See `API_USAGE.md` for detailed API documentation and examples.
