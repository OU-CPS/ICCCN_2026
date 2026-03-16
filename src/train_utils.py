import os
import re
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def load_json_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


# ----------------------------
# Checkpoint utils
# ----------------------------
def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val,
    history,
    extra=None,
    save_optimizer: bool = False
):
    ckpt = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "best_val": float(best_val),
    }

    if save_optimizer and optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()

    if history is not None:
        ckpt["history"] = history

    if extra is not None:
        ckpt["extra"] = extra

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_path = path + ".tmp"
    torch.save(
        ckpt,
        tmp_path,
        _use_new_zipfile_serialization=False
    )
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", np.inf))
    history = ckpt.get("history", {"train": [], "val": []})

    return epoch, best_val, history


# ----------------------------
# LR warmup helpers
# ----------------------------
def set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def linear_warmup_lr(base_lr: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return base_lr
    if epoch <= warmup_epochs:
        return base_lr * float(epoch) / float(warmup_epochs)
    return base_lr


# ----------------------------
# Loss helpers
# ----------------------------
def make_target_alloc(y_true: torch.Tensor, total_center: torch.Tensor, eps: float = 1e-9):
    tc = total_center.unsqueeze(1)
    alloc = y_true / (tc + eps)
    alloc = torch.clamp(alloc, min=0.0)
    s = alloc.sum(dim=1, keepdim=True)
    alloc = alloc / (s + eps)
    return alloc


def sample_weight_from_activity(y_true: torch.Tensor, tau: float, alpha: float):
    active = (y_true > tau).any(dim=1).float()
    w = 1.0 + (alpha - 1.0) * active
    return w


# ----------------------------
# Train
# ----------------------------
def train_transformer_allocation_seq2point_kl(
    model,
    train_loader,
    val_loader,
    device,
    lr,
    epochs,
    patience,
    warmup_epochs,
    ckpt_dir,
    run_name,
    resume_path,
    weight_decay,
    grad_clip,
    kl_weight,
    l1_weight,
    active_tau,
    active_alpha
):
    os.makedirs(ckpt_dir, exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    kl = nn.KLDivLoss(reduction="none")
    l1 = nn.L1Loss(reduction="none")

    history = {"train": [], "val": [], "lr": []}
    best_val = float("inf")
    bad = 0
    start_epoch = 1

    model.to(device)

    if resume_path is not None and os.path.exists(resume_path):
        ep0, best_val0, hist0 = load_checkpoint(resume_path, model, opt, map_location=device)
        start_epoch = ep0 + 1
        best_val = best_val0
        history = hist0
        print(f"Resume from {resume_path}, start_epoch={start_epoch}, best_val={best_val:.6f}")

    best_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")
    last_path = os.path.join(ckpt_dir, f"{run_name}_last.pt")

    eps = 1e-9

    for ep in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()

        current_lr = linear_warmup_lr(base_lr=lr, epoch=ep, warmup_epochs=warmup_epochs)
        set_optimizer_lr(opt, current_lr)
        history["lr"].append(current_lr)

        model.train()
        tr_losses = []

        for xb, yb, tc, hb, db in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            tc = tc.to(device)
            hb = hb.to(device)
            db = db.to(device)

            pred, logits = model(xb, tc, hour_ids=hb, dow_ids=db, return_logits=True)

            mask = (tc > 0).float()
            if mask.sum().item() == 0:
                continue

            target_alloc = make_target_alloc(yb, tc, eps=eps)
            log_alloc = torch.log_softmax(logits, dim=1)

            kl_mat = kl(log_alloc, target_alloc)
            kl_per = kl_mat.sum(dim=1)

            l1_mat = l1(pred, yb)
            l1_per = l1_mat.mean(dim=1)

            w = sample_weight_from_activity(yb, tau=active_tau, alpha=active_alpha)

            loss_per = kl_weight * kl_per + l1_weight * l1_per
            loss = ((loss_per * w) * mask).sum() / (mask.sum() + eps)

            opt.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb, tc, hb, db in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                tc = tc.to(device)
                hb = hb.to(device)
                db = db.to(device)

                pred, logits = model(xb, tc, hour_ids=hb, dow_ids=db, return_logits=True)

                mask = (tc > 0).float()
                if mask.sum().item() == 0:
                    continue

                target_alloc = make_target_alloc(yb, tc, eps=eps)
                log_alloc = torch.log_softmax(logits, dim=1)

                kl_mat = kl(log_alloc, target_alloc)
                kl_per = kl_mat.sum(dim=1)

                l1_mat = l1(pred, yb)
                l1_per = l1_mat.mean(dim=1)

                w = sample_weight_from_activity(yb, tau=active_tau, alpha=active_alpha)

                loss_per = kl_weight * kl_per + l1_weight * l1_per
                loss = ((loss_per * w) * mask).sum() / (mask.sum() + eps)

                va_losses.append(loss.item())

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")
        history["train"].append(tr)
        history["val"].append(va)

        dt = time.perf_counter() - t0
        print(
            f"Epoch {ep:03d} "
            f"lr={current_lr:.8f} "
            f"train_loss={tr:.6f} "
            f"val_loss={va:.6f} "
            f"time={dt:.2f}s"
        )

        save_last_every = 5
        if ep % save_last_every == 0:
            save_checkpoint(
                last_path,
                model,
                opt,
                ep,
                best_val,
                history=None,
                extra=None,
                save_optimizer=False
            )

        if va < best_val - 1e-6:
            best_val = va
            bad = 0
            save_checkpoint(
                best_path,
                model,
                optimizer=None,
                epoch=ep,
                best_val=best_val,
                history=history,
                extra={
                    "lr": lr,
                    "warmup_epochs": warmup_epochs,
                    "kl_weight": kl_weight,
                    "l1_weight": l1_weight
                },
                save_optimizer=False
            )
            print(f"Saved best checkpoint: {best_path} best_val={best_val:.6f}")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if os.path.exists(best_path):
        load_checkpoint(best_path, model, optimizer=None, map_location=device)

    return model, history, best_path, last_path


