"""Registro de clasificadores disponibles.

Cada módulo expone la misma API: train / predict / save / load, más una
constante ``NAME``. Se listan aquí para poder iterarlos por nombre desde
los scripts (``--model all`` o ``--model hog_svm``).
"""
from __future__ import annotations

from . import cnn, color_hist_rf, hog_svm

REGISTRY: dict[str, object] = {
    hog_svm.NAME:        hog_svm,
    color_hist_rf.NAME:  color_hist_rf,
    cnn.NAME:            cnn,
}

# Extensión que usa cada modelo para su archivo guardado
EXTENSIONS: dict[str, str] = {
    hog_svm.NAME:       ".joblib",
    color_hist_rf.NAME: ".joblib",
    cnn.NAME:           ".pt",
}


def get(name: str):
    if name not in REGISTRY:
        opciones = ", ".join(REGISTRY)
        raise KeyError(f"Modelo desconocido: {name!r}. Opciones: {opciones}")
    return REGISTRY[name]
