import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def run_inference_on_loader(model, test_loader, device="cpu", use_time_embedding=True):
    model.eval()
    pred_list = []

    with torch.no_grad():
        for batch in test_loader:
            if use_time_embedding:
                xb, _, tc, hb, db = batch
                xb = xb.to(device)
                tc = tc.to(device)
                hb = hb.to(device)
                db = db.to(device)
                yhat = model(xb, tc, hour_ids=hb, dow_ids=db).cpu().numpy()
            else:
                xb, _, tc = batch
                xb = xb.to(device)
                tc = tc.to(device)
                yhat = model(xb, tc).cpu().numpy()

            pred_list.append(yhat)

    y_pred = np.concatenate(pred_list, axis=0)
    return y_pred


def eval_mae_mape(y_true, y_pred, active_tau: float = 1.0):
    res = {}
    for d in y_true.columns:
        t = y_true[d].values.astype(float)
        p = y_pred[d].values.astype(float)

        err = t - p
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))

        mask = t > active_tau
        if np.any(mask):
            mape = float(np.mean(np.abs(err[mask] / (t[mask] + 1e-9))))
        else:
            mape = float("nan")

        res[d] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    return res





def to_datetime_index(tbin_index, timezone: str = "Australia/Sydney"):
    dt = pd.to_datetime(tbin_index, unit="s", utc=True)
    dt = dt.tz_convert(timezone)
    return dt


def sanitize_filename(name: str) -> str:
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"[^\w\-_\.]", "", s)
    return s

def build_pred_true_dataframes(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    idx_test,
    selected_devices
):
    pred_df = pd.DataFrame(y_pred, index=idx_test, columns=selected_devices)
    true_df = pd.DataFrame(y_true, index=idx_test, columns=selected_devices)
    return pred_df, true_df

def save_pred_true_csvs(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    out_dir: str,
    prefix: str = "test"
):
    os.makedirs(out_dir, exist_ok=True)

    pred_df_to_save = pred_df.copy()
    true_df_to_save = true_df.copy()

    pred_df_to_save.index.name = "tbin"
    true_df_to_save.index.name = "tbin"

    pred_df_to_save.to_csv(os.path.join(out_dir, f"pred_{prefix}_all_devices.csv"))
    true_df_to_save.to_csv(os.path.join(out_dir, f"true_{prefix}_all_devices.csv"))

    for dev in pred_df.columns:
        df_dev = pd.DataFrame({
            "tbin": pred_df.index.astype(np.int64),
            "TRUE": true_df[dev].values.astype(np.float64),
            "pred": pred_df[dev].values.astype(np.float64),
        })
        fname = f"{sanitize_filename(dev)}_true_pred_{prefix}.csv"
        df_dev.to_csv(os.path.join(out_dir, fname), index=False)

def plot_loss_curve(history, save_path="./output/loss_curve.png", title="Training Curve"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_loss = history.get("train", [])
    val_loss = history.get("val", [])
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    
def plot_device_true_pred(
    dev_total,
    pred_df,
    device,
    test_idx,
    tag,
    timezone="Australia/Sydney",
    last_points=None,
    out_dir="./output"
):
    y_true = dev_total.loc[test_idx, device]
    y_pred = pred_df.loc[test_idx, device]

    dt_test = to_datetime_index(y_true.index.values, timezone)

    if (last_points is not None) and (len(y_true) > last_points):
        y_true = y_true.iloc[-last_points:]
        y_pred = y_pred.iloc[-last_points:]
        dt_test = dt_test[-last_points:]

    os.makedirs(out_dir, exist_ok=True)

    dev_s = sanitize_filename(device)
    tag_s = sanitize_filename(tag)

    plt.figure(figsize=(12, 4))
    plt.plot(dt_test, y_true.values, label=f"{device} True", linewidth=1.2)
    plt.plot(dt_test, y_pred.values, label=f"{device} Pred ({tag})", linewidth=1.2, linestyle="--")
    plt.title(f"Device disaggregation on test ({tag}): {device}")
    plt.xlabel("Time")
    plt.ylabel("Bytes per bin")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dev_s}_true_pred_{tag_s}_FULLTEST.png"), dpi=150)
    plt.show()


def plot_device_true_pred_daily(
    dev_total,
    pred_df,
    device,
    test_idx,
    tag,
    timezone="Australia/Sydney",
    out_dir="./output/daily",
    min_points_per_day=1
):
    y_true = dev_total.loc[test_idx, device]
    y_pred = pred_df.loc[test_idx, device]

    dt = to_datetime_index(y_true.index.values, timezone)
    dfp = pd.DataFrame({"true": y_true.values, "pred": y_pred.values}, index=dt).sort_index()

    os.makedirs(out_dir, exist_ok=True)

    dev_s = sanitize_filename(device)
    tag_s = sanitize_filename(tag)

    for day, g in dfp.groupby(pd.Grouper(freq="D")):
        if g.empty or len(g) < min_points_per_day:
            continue

        plt.figure(figsize=(12, 4))
        plt.plot(g.index, g["true"].values, label=f"{device} True", linewidth=1.2)
        plt.plot(g.index, g["pred"].values, label=f"{device} Pred ({tag})", linewidth=1.2, linestyle="--")
        plt.title(f"Device disaggregation on test day ({tag}): {device}  {day.date()}")
        plt.xlabel("Time")
        plt.ylabel("Bytes per bin")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()

        day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
        fname = f"{dev_s}_true_pred_{tag_s}_{day_str}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()


def plot_total_consistency(
    agg_total,
    pred_df,
    idx_test,
    timezone="Australia/Sydney",
    last_points=5000,
    save_path="./output/total_consistency.png"
):
    true_total = agg_total.loc[idx_test]
    pred_total = pred_df.sum(axis=1).reindex(idx_test)
    dt_test = to_datetime_index(true_total.index.values, timezone)

    if len(true_total) > last_points:
        true_total = true_total.iloc[-last_points:]
        pred_total = pred_total.iloc[-last_points:]
        dt_test = dt_test[-last_points:]

    plt.figure(figsize=(12, 4))
    plt.plot(dt_test, true_total.values, label="Total True", linewidth=1.2)
    plt.plot(dt_test, pred_total.values, label="Sum Pred Devices", linewidth=1.2)
    plt.title("Hard conservation check on test")
    plt.xlabel("Time")
    plt.ylabel("Bytes per bin")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()