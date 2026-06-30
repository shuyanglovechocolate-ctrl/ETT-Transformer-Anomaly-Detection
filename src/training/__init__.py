"""Module 3 training and prediction pipeline."""

from src.training.trainer import train_one_epoch, fit_model
from src.training.evaluator import evaluate_loss, compute_metrics
from src.training.predictor import predict, create_prediction_dataframe
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.plots import save_loss_curve, save_prediction_plot

__all__ = [
    "train_one_epoch",
    "fit_model",
    "evaluate_loss",
    "compute_metrics",
    "predict",
    "create_prediction_dataframe",
    "save_checkpoint",
    "load_checkpoint",
    "save_loss_curve",
    "save_prediction_plot",
]
