# ML Prediction REST API

A Flask REST API that serves a scikit-learn classifier, with unit tests (pytest), a Docker image, and a GitHub Actions CI pipeline.

## Endpoints
- `GET /health` -> `{"status": "ok"}`
- `POST /predict` with `{"pixels": [64 numbers, 0-16]}` -> `{"prediction": 3, "confidence": 0.91}`

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # http://localhost:5000
pytest -v                # run tests
```

## Run with Docker
```bash
docker build -t ml-api .
docker run -p 5000:5000 ml-api
```

## Example request
```bash
curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
  -d '{"pixels": [0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]}'
```

## Design notes
- Model trains on first run (or at Docker build) and is cached as `model.joblib`.
- Input is validated; bad requests return HTTP 400 with a clear message.
- CI runs tests on every push, then builds the Docker image.

## Next steps (ideas)
- Replace the digits model with your own (e.g., a deepfake-detection classifier).
- Deploy free on Render or Fly.io and add the live URL here.
