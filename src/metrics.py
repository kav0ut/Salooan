from typing import Mapping, Sequence, Tuple, Union
import numpy as np
import pandas as pd


def _validate_inputs(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
    alpha: Union[float, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(y_true, pd.Series):
        y_true_arr = y_true.to_numpy(dtype=np.float64, copy=True)
    else:
        y_true_arr = np.asarray(y_true, dtype=np.float64)

    if isinstance(y_pred, pd.Series):
        y_pred_arr = y_pred.to_numpy(dtype=np.float64, copy=True)
    else:
        y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        y_true_arr = y_true_arr.ravel()
        y_pred_arr = y_pred_arr.ravel()

    if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
        raise ValueError("Inputs cannot be empty.")

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true_arr)} != len(y_pred)={len(y_pred_arr)}."
        )

    if np.isnan(y_true_arr).any() or np.isnan(y_pred_arr).any():
        raise ValueError("Inputs cannot contain NaN values.")

    if np.isinf(y_true_arr).any() or np.isinf(y_pred_arr).any():
        raise ValueError("Inputs cannot contain infinite values.")

    if alpha is not None:
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError(f"alpha must be a numeric float, got {type(alpha).__name__}.")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}.")

    return y_true_arr, y_pred_arr


def pinball_loss(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
    alpha: float,
) -> float:
    y_arr, q_arr = _validate_inputs(y_true, y_pred, alpha=alpha)
    diff = y_arr - q_arr
    # Pinball loss: max(alpha * (y - q), (alpha - 1) * (y - q))
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


def breach_rate(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
) -> float:
    y_arr, q_arr = _validate_inputs(y_true, y_pred)
    # Strict inequality: realized return is worse than predicted quantile
    return float(np.mean(y_arr < q_arr))


def crossing_rate(
    predictions_by_confidence: Mapping[Union[float, str], Union[np.ndarray, pd.Series, Sequence[float]]],
) -> float:
    if len(predictions_by_confidence) < 2:
        raise ValueError("crossing_rate requires at least two confidence levels.")

    sorted_confs = sorted(predictions_by_confidence.keys(), key=float)
    arrays = []
    expected_len = None

    for conf in sorted_confs:
        val = predictions_by_confidence[conf]
        if isinstance(val, pd.Series):
            arr = val.to_numpy(dtype=np.float64, copy=True)
        else:
            arr = np.asarray(val, dtype=np.float64)

        if arr.ndim != 1:
            arr = arr.ravel()

        if len(arr) == 0:
            raise ValueError("Prediction arrays cannot be empty.")

        if expected_len is None:
            expected_len = len(arr)
        elif len(arr) != expected_len:
            raise ValueError(f"Length mismatch: expected {expected_len}, got {len(arr)}.")

        if np.isnan(arr).any():
            raise ValueError("Prediction arrays cannot contain NaN values.")

        if np.isinf(arr).any():
            raise ValueError("Prediction arrays cannot contain infinite values.")

        arrays.append(arr)

    matrix = np.column_stack(arrays)
    # Predicted returns must be non-increasing with confidence; diff > 0 indicates crossing
    diffs = np.diff(matrix, axis=1)
    violating_rows = np.any(diffs > 0, axis=1)
    return float(np.mean(violating_rows))
