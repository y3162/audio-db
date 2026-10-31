"""Evaluation pipeline for speech enhancement.

Stages (dataset, mixing, enhancement, recognition, metrics, results) are
independent. Speech enhancement models are loaded with src.se.load_enhancer.
This package does not train models and does not modify the corpus database.
"""

from src.evaluation.identity import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
