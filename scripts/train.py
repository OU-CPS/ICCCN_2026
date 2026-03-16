import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader



sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_utils import build_dev_series_selected, align_up_down
from src.features import (
    build_input_4ch,
    make_seq2point_arrays_two_aggs_with_time,
    transform_standardize_4ch,
)
from src.datasets import Seq2PointDatasetWithTotalTime
from src.model_transformer import Seq2PointTransformerAllocation
from src.train_utils import (
    load_json_config,
    train_transformer_allocation_seq2point_kl,
)

from src.eval_utils import (
    eval_mae_mape,
    plot_loss_curve,
    plot_device_true_pred,
    plot_device_true_pred_daily,
    plot_total_consistency,
    run_inference_on_loader,
    build_pred_true_dataframes,
    save_pred_true_csvs,
)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config.json")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_json_config(args.config)

    selected_devices = cfg["selected_devices"]

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    out_cfg = cfg["output"]

    up_dir = data_cfg["up_dir"]
    down_dir = data_cfg["down_dir"]
    bin_seconds = data_cfg["bin_seconds"]
    window = data_cfg["window"]
    stride = data_cfg["stride"]
    timezone = data_cfg["timezone"]
    keep_days = data_cfg["keep_days"]
    train_ratio = data_cfg["train_ratio"]
    val_ratio_within_train = data_cfg["val_ratio_within_train"]

    output_dir = out_cfg["output_dir"]
    run_name = out_cfg["run_name"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # ----------------------------
    # Build data
    # ----------------------------
    dev_up = build_dev_series_selected(
        device_dir=up_dir,
        selected_devices=selected_devices,
        bin_seconds=bin_seconds,
        suffix="uplink.csv",
        keep_days=keep_days
    )

    dev_down = build_dev_series_selected(
        device_dir=down_dir,
        selected_devices=selected_devices,
        bin_seconds=bin_seconds,
        suffix="downlink.csv",
        keep_days=keep_days
    )

    dev_up, dev_down = align_up_down(dev_up, dev_down)

    agg_up = dev_up.sum(axis=1)
    agg_down = dev_down.sum(axis=1)

    print("dev_up shape:", dev_up.shape)
    print("dev_down shape:", dev_down.shape)

    # ----------------------------
    # Build windows with time ids
    # ----------------------------
    X_up, X_down, HOUR, DOW, Y, out_index, center_total_up_all, r = \
        make_seq2point_arrays_two_aggs_with_time(
            agg_up=agg_up,
            agg_down=agg_down,
            dev_target=dev_up,
            window_length=window,
            stride=stride,
            timezone=timezone
        )

    cut = int(len(out_index) * train_ratio)

    X_up_train, X_up_test = X_up[:cut], X_up[cut:]
    X_down_train, X_down_test = X_down[:cut], X_down[cut:]

    H_train, H_test = HOUR[:cut], HOUR[cut:]
    D_train, D_test = DOW[:cut], DOW[cut:]

    Y_train, Y_test = Y[:cut], Y[cut:]
    idx_train, idx_test = out_index[:cut], out_index[cut:]
    center_total_train, center_total_test = center_total_up_all[:cut], center_total_up_all[cut:]

    # ----------------------------
    # Build 4-channel input
    # ----------------------------
    X4_train = build_input_4ch(X_up_train, X_down_train)
    X4_test = build_input_4ch(X_up_test, X_down_test)

    X4_train_std, X4_test_std, mu, sd = transform_standardize_4ch(X4_train, X4_test)

    # ----------------------------
    # Train/val split
    # ----------------------------
    val_cut = int(len(idx_train) * val_ratio_within_train)

    X_tr = X4_train_std[:val_cut]
    Y_tr = Y_train[:val_cut]
    TC_tr = center_total_train[:val_cut]
    H_tr = H_train[:val_cut]
    D_tr = D_train[:val_cut]

    X_va = X4_train_std[val_cut:]
    Y_va = Y_train[val_cut:]
    TC_va = center_total_train[val_cut:]
    H_va = H_train[val_cut:]
    D_va = D_train[val_cut:]

    print("Input dim:", X_tr.shape[2])
    print("Y_train min:", Y_train.min(), "max:", Y_train.max())
    print("Hour ID range:", HOUR.min(), HOUR.max())
    print("DOW ID range:", DOW.min(), DOW.max())

    # ----------------------------
    # DataLoader
    # ----------------------------
    train_loader = DataLoader(
        Seq2PointDatasetWithTotalTime(X_tr, Y_tr, TC_tr, H_tr, D_tr),
        batch_size=train_cfg["batch_size_train"],
        shuffle=True
    )
    val_loader = DataLoader(
        Seq2PointDatasetWithTotalTime(X_va, Y_va, TC_va, H_va, D_va),
        batch_size=train_cfg["batch_size_eval"],
        shuffle=False
    )
    test_loader = DataLoader(
        Seq2PointDatasetWithTotalTime(X4_test_std, Y_test, center_total_test, H_test, D_test),
        batch_size=train_cfg["batch_size_eval"],
        shuffle=False
    )

    # ----------------------------
    # Model
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = Seq2PointTransformerAllocation(
        input_dim=model_cfg["input_dim"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_ff=model_cfg["dim_ff"],
        n_outputs=len(selected_devices),
        center_idx=window // 2,
        dropout=model_cfg["dropout"],
        max_len=window,
        norm_first=model_cfg["norm_first"],
        use_time_embedding=model_cfg["use_time_embedding"]
    )

    # ----------------------------
    # Train
    # ----------------------------
    model, history, best_path, last_path = train_transformer_allocation_seq2point_kl(
        model,
        train_loader,
        val_loader,
        device=device,
        lr=train_cfg["lr"],
        epochs=train_cfg["epochs"],
        patience=train_cfg["patience"],
        warmup_epochs=train_cfg["warmup_epochs"],
        ckpt_dir=os.path.join(output_dir, "checkpoints"),
        run_name=run_name,
        resume_path=train_cfg["resume_path"],
        weight_decay=train_cfg["weight_decay"],
        grad_clip=train_cfg["grad_clip"],
        kl_weight=train_cfg["kl_weight"],
        l1_weight=train_cfg["l1_weight"],
        active_tau=train_cfg["active_tau"],
        active_alpha=train_cfg["active_alpha"]
    )

    plot_loss_curve(
        history,
        save_path=os.path.join(output_dir, f"loss_curve_{run_name}.png"),
        title="Seq2Point Transformer Allocation KL Training Curve"
    )

    # ----------------------------
    # Inference
    # ----------------------------
    Y_pred_test = run_inference_on_loader(
        model=model,
        test_loader=test_loader,
        device=device,
        use_time_embedding=model_cfg["use_time_embedding"]
    )

    

    # ----------------------------
    # Eval
    # ----------------------------
    pred_tx_alloc, true_test = build_pred_true_dataframes(
        y_pred=Y_pred_test,
        y_true=Y_test,
        idx_test=idx_test,
        selected_devices=selected_devices
    )

    pred_tx_alloc.to_csv(os.path.join(output_dir, f"{run_name}_pred.csv"))
    true_test.to_csv(os.path.join(output_dir, f"{run_name}_true.csv"))

    save_pred_true_csvs(
        pred_df=pred_tx_alloc,
        true_df=true_test,
        out_dir=os.path.join(output_dir, "per_device_csv"),
        prefix="test"
    )
    
    sum_pred = pred_tx_alloc.sum(axis=1)
    true_total = agg_up.loc[idx_test]
    print("MAE(uplink total conservation):", float(np.mean(np.abs(sum_pred.values - true_total.values))))

    metrics = eval_mae_mape(true_test, pred_tx_alloc, active_tau=1.0 * bin_seconds)
    print("\nTransformer allocation KL metrics on test")
    for d, m in metrics.items():
        print(d, m)

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(output_dir, f"{run_name}_metrics.csv"))

    # ----------------------------
    # Plots
    # ----------------------------
    full_dir = os.path.join(output_dir, "test_full")
    daily_dir = os.path.join(output_dir, "test_daily")

    for dev in selected_devices:
        plot_device_true_pred(
            dev_total=dev_up,
            pred_df=pred_tx_alloc,
            device=dev,
            test_idx=idx_test,
            tag=run_name,
            timezone=timezone,
            last_points=None,
            out_dir=full_dir
        )

        plot_device_true_pred_daily(
            dev_total=dev_up,
            pred_df=pred_tx_alloc,
            device=dev,
            test_idx=idx_test,
            tag=run_name,
            timezone=timezone,
            out_dir=daily_dir,
            min_points_per_day=1
        )

    plot_total_consistency(
        agg_total=agg_up,
        pred_df=pred_tx_alloc,
        idx_test=idx_test,
        timezone=timezone,
        last_points=100000,
        save_path=os.path.join(output_dir, f"total_consistency_{run_name}.png")
    )


if __name__ == "__main__":
    main()
