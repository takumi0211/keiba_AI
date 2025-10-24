"""
Model training utilities for the 菊花賞モデル project.

Currently exposes the LightGBM training pipeline.
"""

from .light_gbm import TrainingArtifacts, calc_topk_hits, train_model

__all__ = ["TrainingArtifacts", "calc_topk_hits", "train_model"]
