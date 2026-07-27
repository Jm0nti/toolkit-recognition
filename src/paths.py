"""Rutas comunes ancladas a la raíz del repositorio.

Todo módulo/script del proyecto debería importar rutas desde aquí en vez de
usar strings relativos al cwd, para que se puedan ejecutar desde cualquier
directorio de trabajo.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path         = REPO_ROOT / "data"
DATA_RAW_DIR: Path     = DATA_DIR / "raw"
DATA_AUG_DIR: Path     = DATA_DIR / "augmented"
UNIFIED_NDJSON: Path   = DATA_RAW_DIR / "unified.ndjson"

MODELS_DIR: Path         = REPO_ROOT / "models"
DETECTION_MODELS: Path   = MODELS_DIR / "detection"
CLASSIF_MODELS: Path     = MODELS_DIR / "classification"

OUTPUTS_DIR: Path        = REPO_ROOT / "outputs"
PREDICTIONS_DIR: Path    = OUTPUTS_DIR / "predicciones"
ML_DATASETS_DIR: Path    = OUTPUTS_DIR / "ml_datasets"
ML_REPORTS_DIR: Path     = OUTPUTS_DIR / "ml_reports"

RUNS_DIR: Path         = REPO_ROOT / "runs"
WEB_DIR: Path          = REPO_ROOT / "web"
DATA_YAML: Path        = REPO_ROOT / "data.yaml"
