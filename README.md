# Brain Tumor Detection Dashboard

FastAPI backend + static frontend for the supplied NeuroScan AI brain MRI classification implementation.

## Structure

- `backend/main.py` — FastAPI API and static frontend serving
- `backend/model_loader.py` — model loading and CBAM custom layers
- `backend/gradcam.py` — Grad-CAM generation
- `backend/models/` — trained model files
- `frontend/index.html` — dashboard UI
- `frontend/style.css` — dashboard styling
- `frontend/script.js` — upload/prediction frontend logic
- `save_model.py` — notebook snippet for saving the trained model

## Run

From the project root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then open:

http://localhost:8000

## Required model files

Copy these into `backend/models/`:

- `brain_tumor_final.keras`
- `class_names.json`
- `config.json`

The trained model binary is not included because it was not present in the supplied text.



cd C:\Users\amank\Downloads\brain_tumor_dashboard
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000