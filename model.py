"""Train and load a small scikit-learn classifier (swap in your own model/data later)."""
import joblib
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).parent / "model.joblib"


def train(path: Path = MODEL_PATH) -> float:
    """Train on the bundled digits dataset, save the model, return test accuracy."""
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    joblib.dump(clf, path)
    return clf.score(X_te, y_te)


def load(path: Path = MODEL_PATH):
    if not path.exists():
        train(path)
    return joblib.load(path)
