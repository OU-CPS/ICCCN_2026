import numpy as np
import pandas as pd


def build_time_features_from_index(tbin_index, timezone: str = "Australia/Sydney"):
    dt = pd.to_datetime(tbin_index, unit="s", utc=True).tz_convert(timezone)
    hour_arr = dt.hour.astype(np.int64).to_numpy()
    dow_arr = dt.dayofweek.astype(np.int64).to_numpy()
    return hour_arr, dow_arr


def signed_log1p(a: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.log1p(np.abs(a))


def build_input_4ch(up_windows: np.ndarray, down_windows: np.ndarray) -> np.ndarray:
    up = up_windows
    down = down_windows
    up_delta = np.diff(up, axis=1, prepend=up[:, :1])
    down_delta = np.diff(down, axis=1, prepend=down[:, :1])
    X4 = np.stack([up, up_delta, down, down_delta], axis=2)
    return X4


def make_seq2point_arrays_two_aggs_with_time(
    agg_up: pd.Series,
    agg_down: pd.Series,
    dev_target: pd.DataFrame,
    window_length: int = 599,
    stride: int = 1,
    timezone: str = "Australia/Sydney"
):
    assert window_length % 2 == 1
    L = window_length
    r = L // 2

    x_up = agg_up.values.astype(np.float32)
    x_down = agg_down.values.astype(np.float32)
    y = dev_target.values.astype(np.float32)
    idx = agg_up.index.values

    hour_arr, dow_arr = build_time_features_from_index(idx, timezone=timezone)

    T = len(x_up)
    centers = np.arange(r, T - r, stride)

    X_up = np.stack([x_up[c - r:c + r + 1] for c in centers], axis=0)
    X_down = np.stack([x_down[c - r:c + r + 1] for c in centers], axis=0)

    HOUR = np.stack([hour_arr[c - r:c + r + 1] for c in centers], axis=0).astype(np.int64)
    DOW = np.stack([dow_arr[c - r:c + r + 1] for c in centers], axis=0).astype(np.int64)

    Y = y[centers, :]
    out_index = idx[centers]
    center_total_up = X_up[:, r].copy()

    return X_up, X_down, HOUR, DOW, Y, out_index, center_total_up, r


def transform_standardize_4ch(X4_train: np.ndarray, X4_test: np.ndarray):
    X4_train_t = X4_train.copy()
    X4_test_t = X4_test.copy()

    X4_train_t[:, :, 0] = np.log1p(np.maximum(X4_train_t[:, :, 0], 0.0))
    X4_test_t[:, :, 0] = np.log1p(np.maximum(X4_test_t[:, :, 0], 0.0))

    X4_train_t[:, :, 1] = signed_log1p(X4_train_t[:, :, 1])
    X4_test_t[:, :, 1] = signed_log1p(X4_test_t[:, :, 1])

    X4_train_t[:, :, 2] = np.log1p(np.maximum(X4_train_t[:, :, 2], 0.0))
    X4_test_t[:, :, 2] = np.log1p(np.maximum(X4_test_t[:, :, 2], 0.0))

    X4_train_t[:, :, 3] = signed_log1p(X4_train_t[:, :, 3])
    X4_test_t[:, :, 3] = signed_log1p(X4_test_t[:, :, 3])

    mu = X4_train_t.mean(axis=0, keepdims=True)
    sd = X4_train_t.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-3)

    X4_train_std = (X4_train_t - mu) / sd
    X4_test_std = (X4_test_t - mu) / sd

    return X4_train_std, X4_test_std, mu, sd