"""Clasificador HOG + SVM (kernel RBF).

  * Extracción: HOG sobre la imagen en gris (128×128 tras padding).
  * Estandarización: StandardScaler para que el SVM converja bien.
  * Modelo: SVC (RBF, C=10, gamma='scale'). Robusto para descriptores de textura.

Interfaz común a los tres clasificadores:
    train(X_bgr, y, class_names) -> model
    predict(model, X_bgr)         -> np.ndarray
    save(model, path)             -> None
    load(path)                    -> model
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..features import hog_features


NAME = "hog_svm"


def _extract(X_bgr: list[np.ndarray]) -> np.ndarray:
    return np.vstack([hog_features(img) for img in X_bgr])


def train(X_bgr: list[np.ndarray],
          y: np.ndarray,
          class_names: list[str] | None = None,
          C: float = 10.0,
          gamma: str = "scale") -> dict:
    """Entrena el pipeline StandardScaler + SVC(RBF)."""
    print(f"[{NAME}] Extrayendo HOG de {len(X_bgr)} recortes...")
    t0 = time.time()
    X_feat = _extract(X_bgr)
    t_feat = time.time() - t0
    print(f"[{NAME}] HOG listo: shape={X_feat.shape}  ({t_feat:.1f}s)")

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svc",    SVC(kernel="rbf", C=C, gamma=gamma, cache_size=500)),
    ])
    t0 = time.time()
    clf.fit(X_feat, y)
    t_train = time.time() - t0
    print(f"[{NAME}] SVM entrenado en {t_train:.1f}s")

    return {"clf": clf, "class_names": class_names, "duracion_train_s": t_feat + t_train}


def predict(model: dict, X_bgr: list[np.ndarray]) -> np.ndarray:
    X_feat = _extract(X_bgr)
    return model["clf"].predict(X_feat)


def save(model: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[{NAME}] guardado en {path}")


def load(path: str | Path) -> dict:
    return joblib.load(path)
