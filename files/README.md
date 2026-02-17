# CatfishGuard – AI-Powered Catfishing Detection Tool

**GWC Challenge 2025-26: Cybersecurity + AI**

CatfishGuard helps users detect AI-generated profile pictures used in catfishing. Upload a screenshot of a suspicious profile picture and get an instant AI-powered risk assessment.

## How It Works

The tool analyzes uploaded images across **7 AI detection layers**:

1. **Anatomical Anomalies** – Detects warped ears, twisted fingers, mismatched eyes, distorted jawlines
2. **Background Coherence** – Spots warped lines, melting objects, incoherent spatial relationships
3. **"Too Perfect" Skin** – Identifies glassy, airbrushed skin lacking pores and natural micro-texture
4. **Text & Object Artifacts** – Catches garbled text, scrambled letters, broken object boundaries
5. **Lighting Inconsistencies** – Detects mismatched shadow directions and lighting between subject/background
6. **Frequency Domain Analysis** – Examines frequency signatures that real cameras produce but AI cannot
7. **Noise Pattern Analysis** – Checks for natural sensor noise that varies with brightness

## Setup & Run

### Requirements
- Python 3.8+
- pip

### Installation

```bash
pip install flask pillow numpy
python app.py
```

Then open **http://localhost:5000** in your browser.

### Project Structure
```
catfish-detector/
├── app.py                 # Flask backend with 7 AI analysis layers
├── templates/
│   └── index.html         # Main page template
├── static/
│   ├── style.css          # Styles (dark cyber theme)
│   └── script.js          # Frontend logic
├── uploads/               # Temporary upload directory
└── README.md
```

## Tech Stack
- **Backend:** Python, Flask, Pillow, NumPy
- **Frontend:** HTML5, CSS3, JavaScript (vanilla)
- **AI Techniques:** Anatomical distortion detection, background coherence analysis, skin micro-texture analysis, text/object artifact detection, lighting direction consistency, FFT frequency domain, noise pattern analysis

## Disclaimer
This tool provides AI-based probabilistic analysis and is not 100% accurate. Always use multiple verification methods.
