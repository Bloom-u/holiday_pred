from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), 1.0)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def compute_metrics(y_true, y_pred):
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mape=float(mape(y_true, y_pred)),
    )
