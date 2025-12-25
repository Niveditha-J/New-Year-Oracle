# New Year Oracle 

A small, playful Streamlit app that "pretends" to predict your 2026 based on simple face heuristics and randomized storytelling. For entertainment purposes only.

## What it does
- Uses your webcam to capture a frame.
- Runs a lightweight heuristic over face landmarks (via MediaPipe) to estimate mood.
- Generates fun, weighted predictions and reveals them with a dramatic UI.

## Install & run

1. Create a virtual environment (recommended) and install:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
\.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

Open the served page in your browser, allow camera access, and try it out.

## Notes & ethics
- This project uses simple heuristics and randomized outputs — it does not predict the future.
- Tiny disclaimer is shown in the UI: *For entertainment purposes only.*

## Files
- `app.py` — Streamlit UI
- `predictor.py` — heuristic analysis + prediction engine
- `requirements.txt` — Python deps

Have fun! 🎉
