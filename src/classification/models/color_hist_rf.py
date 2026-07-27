"""Clasificador Histograma de color (HSV) + Random Forest.

  * Extracción: histograma 3D HSV con bins (8,8,8) normalizado.
  * Modelo: RandomForestClassifier(n_estimators=200, max_depth=None).

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
from sklearn.ensemble import RandomForestClassifier

from ..features import color_histogram


NAME = "color_hist_rf"


def _extract(X_bgr: list[np.ndarray]) -> np.ndarray:
    return np.vstack([color_histogram(img) for img in X_bgr])


def train(X_bgr: list[np.ndarray],
          y: np.ndarray,
          class_names: list[str] | None = None,
          n_estimators: int = 200,
          max_depth: int | None = None,
          n_jobs: int = -1) -> dict:
    """Entrena RandomForest sobre histogramas HSV."""
    print(f"[{NAME}] Extrayendo histogramas HSV de {len(X_bgr)} recortes...")
    t0 = time.time()
    X_feat = _extract(X_bgr)
    t_feat = time.time() - t0
    print(f"[{NAME}] Histogramas listos: shape={X_feat.shape}  ({t_feat:.1f}s)")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=42,
        class_weight="balanced",
    )
    t0 = time.time()
    clf.fit(X_feat, y)
    t_train = time.time() - t0
    print(f"[{NAME}] RandomForest entrenado en {t_train:.1f}s")

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
