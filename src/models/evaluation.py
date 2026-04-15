"""Model evaluation utilities."""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute evaluation metrics for regression and classification models."""

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Return regression metrics: rmse, mae, r2, mape."""
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        nonzero = y_true != 0
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float('nan')

        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Return classification metrics: accuracy, precision, recall, f1, roc_auc, etc."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        result: Dict[str, Any] = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            y_proba = np.array(y_proba)
            classes = np.unique(y_true)
            if len(classes) == 2:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                result['roc_auc'] = float(roc_auc_score(y_true, proba_pos))
                fpr_vals = np.linspace(0, 1, 100)
                from sklearn.metrics import roc_curve
                fpr, tpr, thresholds = roc_curve(y_true, proba_pos)
                result['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                }
            else:
                result['roc_auc'] = float(
                    roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                )

        return result

    def profit_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        cost_benefit_matrix: Optional[np.ndarray] = None,
        n_thresholds: int = 100,
    ) -> Dict[str, List[float]]:
        """Compute profit curve across classification thresholds."""
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)

        if cost_benefit_matrix is None:
            cost_benefit_matrix = np.array([[1, -1], [-1, 0]])

        thresholds = np.linspace(0, 1, n_thresholds)
        profits = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            profit = float(np.sum(cm * cost_benefit_matrix) / len(y_true))
            profits.append(profit)

        return {
            'thresholds': thresholds.tolist(),
            'profits': profits,
        }
