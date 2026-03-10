"""
evaluation/metrics.py
======================
Runs the trained model against the test set and computes all required
evaluation metrics and plots.

Primary metric: Sensitivity (recall for the melanoma class, label=1).
    False negatives (missed melanomas) are more harmful than false positives
    in a screening context, so sensitivity is prioritized over accuracy.

Required outputs:
    - Confusion matrix (saved as PNG)
    - Sensitivity  — true positive rate for melanoma
    - Precision    — positive predictive value
    - F1 score     — harmonic mean of precision and sensitivity
    - ROC-AUC      — area under the receiver operating characteristic curve
    - Specificity  — true negative rate (non-melanoma correctly identified)

All plots are saved to the eval_output_dir specified in config.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_predictions(model: keras.Model, test_ds: tf.data.Dataset) -> tuple:
    """
    Run inference on the full test set and collect outputs.

    Uses model.predict() across all batches. The test dataset must not
    be shuffled so that ground-truth labels stay aligned with predictions.

    Args:
        model (keras.Model):        Trained model with sigmoid output.
        test_ds (tf.data.Dataset):  Test split, batched but not shuffled.

    Returns:
        tuple:
            y_pred_proba (np.ndarray): Sigmoid probabilities, shape (N,).
            y_true (np.ndarray):       Ground truth labels (0 or 1), shape (N,).
    """
    pass


def apply_threshold(y_pred_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert sigmoid probabilities to binary class predictions.

    Lowering the threshold below 0.5 increases sensitivity at the
    cost of more false positives (lower precision). The threshold can
    be tuned post-training using the ROC curve.

    Args:
        y_pred_proba (np.ndarray): Predicted probabilities, shape (N,).
        threshold (float):         Decision boundary. Default 0.5.

    Returns:
        np.ndarray: Binary predictions (0 or 1), shape (N,).
    """
    pass


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the 2×2 confusion matrix for binary classification.

    Matrix layout:
        [[TN, FP],
         [FN, TP]]

    Args:
        y_true (np.ndarray): Ground truth labels (0 or 1), shape (N,).
        y_pred (np.ndarray): Binary predictions (0 or 1), shape (N,).

    Returns:
        np.ndarray: 2×2 confusion matrix of integer counts.
    """
    pass


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    """
    Compute all required evaluation metrics from predictions.

    Uses sklearn.metrics internally. Specificity is derived from the
    confusion matrix as TN / (TN + FP).

    Args:
        y_true (np.ndarray):       Ground truth labels, shape (N,).
        y_pred (np.ndarray):       Binary predictions, shape (N,).
        y_pred_proba (np.ndarray): Raw sigmoid probabilities, shape (N,).

    Returns:
        dict: {
            'sensitivity': float,   # recall for melanoma (class 1)
            'specificity': float,   # recall for non-melanoma (class 0)
            'precision':   float,
            'f1_score':    float,
            'roc_auc':     float,
            'accuracy':    float,
        }
    """
    pass


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    normalize: bool = True,
) -> None:
    """
    Save a confusion matrix heatmap as a PNG file.

    Uses matplotlib + seaborn. Axis labels are ['Non-Melanoma', 'Melanoma'].
    When normalize=True, displays row percentages instead of raw counts.

    Args:
        cm (np.ndarray):    2×2 confusion matrix from compute_confusion_matrix().
        output_path (str):  Destination path for the PNG file.
        normalize (bool):   Display percentages if True, counts if False.

    Returns:
        None. Writes PNG to output_path.
    """
    pass


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: str,
) -> None:
    """
    Save a ROC curve plot (TPR vs FPR) with AUC annotation as a PNG.

    Includes a diagonal reference line representing a random classifier.
    Title displays the computed AUC value.

    Args:
        y_true (np.ndarray):       Ground truth labels, shape (N,).
        y_pred_proba (np.ndarray): Sigmoid probabilities, shape (N,).
        output_path (str):         Destination path for the PNG file.

    Returns:
        None. Writes PNG to output_path.
    """
    pass


def evaluate(model: keras.Model, test_ds: tf.data.Dataset, config: dict) -> dict:
    """
    Top-level evaluation function — runs the full evaluation pipeline.

    Calls in order:
        1. get_predictions()          → raw probabilities + ground truth
        2. apply_threshold()          → binary predictions
        3. compute_confusion_matrix() → CM array
        4. compute_metrics()          → metrics dict
        5. plot_confusion_matrix()    → saves CM plot
        6. plot_roc_curve()           → saves ROC plot
        7. Prints formatted metrics summary to console
        8. Writes metrics dict to JSON for reproducibility

    Args:
        model (keras.Model):        Trained model.
        test_ds (tf.data.Dataset):  Test split, not shuffled.
        config (dict):              Must contain:
                                    - evaluation.threshold
                                    - paths.eval_output_dir

    Returns:
        dict: Metrics dictionary from compute_metrics().
    """
    pass