#!/usr/bin/env python3
"""Wrapper para src/data/merge_ndjson.py.

Ejecuta la unificación de los .ndjson de los colaboradores usando la
configuración definida en src/data/merge_ndjson.py (INPUT_FILES/OUTPUT_FILE).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.merge_ndjson import main  # noqa: E402


if __name__ == "__main__":
    main()
