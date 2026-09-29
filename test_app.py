import pytest
from sklearn.datasets import load_digits
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_predict_valid(client):
    X, y = load_digits(return_X_y=True)
    r = client.post("/predict", json={"pixels": X[0].tolist()})
    assert r.status_code == 200
    body = r.get_json()
    assert 0 <= body["prediction"] <= 9
    assert 0 <= body["confidence"] <= 1


def test_predict_missing_field(client):
    assert client.post("/predict", json={}).status_code == 400


def test_predict_wrong_length(client):
    assert client.post("/predict", json={"pixels": [1, 2, 3]}).status_code == 400


def test_predict_bad_types(client):
    assert client.post("/predict", json={"pixels": ["a"] * 64}).status_code == 400
