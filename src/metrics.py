from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {
        "r2":   round(r2,   4),
        "mae":  round(mae,  2),
        "rmse": round(rmse, 2),
    }
