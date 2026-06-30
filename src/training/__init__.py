"""Module 3 training and prediction pipeline."""

from src.training.trainer import train_one_epoch, fit_model, build_scheduler
from src.training.evaluator import evaluate_loss, compute_metrics
from src.training.predictor import predict, create_prediction_dataframe
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.plots import save_loss_curve, save_prediction_plot
from src.training.early_stopping import EarlyStopping

__all__ = [
    "train_one_epoch",
    "fit_model",
    "build_scheduler",
    "evaluate_loss",
    "compute_metrics",
    "predict",
    "create_prediction_dataframe",
    "save_checkpoint",
    "load_checkpoint",
    "save_loss_curve",
    "save_prediction_plot",
    "EarlyStopping",
]
