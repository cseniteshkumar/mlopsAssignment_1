"""Tiny runner script to execute the pipeline CLI.

Usage:
    python run_pipeline.py --dataset DataSet --output models/model.joblib
"""
import sys
from pathlib import Path

# Ensure repository root is on sys.path so `src` imports resolve when called from project root
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline()
