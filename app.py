from flask import Flask, jsonify, request
import model as model_module

app = Flask(__name__)
_clf = None


def get_model():
    global _clf
    if _clf is None:
        _clf = model_module.load()
    return _clf


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data or "pixels" not in data:
        return jsonify(error="JSON body must include 'pixels'"), 400
    pixels = data["pixels"]
    if (not isinstance(pixels, list) or len(pixels) != 64
            or not all(isinstance(p, (int, float)) for p in pixels)):
        return jsonify(error="'pixels' must be a list of 64 numbers (8x8 image, values 0-16)"), 400
    clf = get_model()
    probs = clf.predict_proba([pixels])[0]
    label = int(probs.argmax())
    return jsonify(prediction=label, confidence=round(float(probs[label]), 4))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
