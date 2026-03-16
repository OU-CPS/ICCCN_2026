import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, matthews_corrcoef, f1_score


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_filename(name: str) -> str:
    s = str(name).strip().replace(" ", "_")
    return re.sub(r"[^\w\_\.]", "", s)


def aggregate_packets_to_bins(packet_csv, bin_seconds=1, keep_days=None):
    df = pd.read_csv(packet_csv)
    if keep_days is not None and "day" in df.columns:
        df = df[df["day"].isin(keep_days)]

    df["TIME"] = pd.to_numeric(df["TIME"], errors="coerce")
    df["Size"] = pd.to_numeric(df["Size"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["TIME"])
    df.loc[df["Size"] < 0, "Size"] = 0.0

    t = df["TIME"].astype(np.int64)
    df["tbin"] = (t // bin_seconds) * bin_seconds
    return df.groupby("tbin")["Size"].sum().sort_index().astype(np.float64)


def complete_index(ts, bin_seconds=1):
    if ts.empty:
        return ts
    t0, t1 = int(ts.index.min()), int(ts.index.max())
    full = np.arange(t0, t1 + bin_seconds, bin_seconds, dtype=np.int64)
    return ts.reindex(full).fillna(0.0).astype(np.float64)


SEMANTIC_LABELS_TOTAL = {
    2: ["idle", "active"],
    3: ["idle", "low", "active"],
    4: ["idle", "low", "medium", "high"],
    5: ["idle", "background", "low", "medium", "high"],
    6: ["idle", "background", "low", "medium", "high", "burst"],
}


def evaluate_cluster_validity(cluster_vals_list):
    k = len(cluster_vals_list)
    stats = []
    for vals in cluster_vals_list:
        if vals.size == 0:
            stats.append({"mean": 0, "std": 0, "p5": 0, "p95": 0, "count": 0})
            continue
        stats.append(
            {
                "mean": float(vals.mean()),
                "std": float(vals.std()) if vals.size > 1 else 0.0,
                "p5": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
                "count": int(vals.size),
            }
        )

    validity = []
    for i in range(k):
        s = stats[i]
        width = s["p95"] - s["p5"] if s["count"] > 1 else 1.0
        width = max(float(width), 1.0)

        gaps = []
        if i > 0:
            gaps.append(s["p5"] - stats[i - 1]["p95"])
        if i < k - 1:
            gaps.append(stats[i + 1]["p5"] - s["p95"])

        min_gap = float(min(gaps)) if gaps else 0.0
        sep_ratio = float(min_gap / width) if width > 0 else 0.0
        cv = float(s["std"] / s["mean"]) if s["mean"] > 0 else 0.0

        validity.append(
            {
                "cluster_rank": i,
                "count": s["count"],
                "mean": s["mean"],
                "std": s["std"],
                "width_p5_p95": width,
                "min_gap_to_neighbor": min_gap,
                "separation_ratio": sep_ratio,
                "cv": cv,
                "is_separable": (min_gap > 0) or (s["count"] >= 10),
            }
        )
    return validity


def _stratified_sample_indices(labels, sample_n=50000, min_per_cluster=50, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    sample_n = min(n, int(sample_n))
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return None

    chosen = []
    remaining = sample_n
    for cl in uniq:
        idx = np.where(labels == cl)[0]
        take = min(len(idx), int(min_per_cluster))
        if take > 0:
            chosen.append(rng.choice(idx, take, replace=False))
            remaining -= take

    if remaining > 0:
        already = np.concatenate(chosen) if chosen else np.array([], dtype=int)
        mask = np.ones(n, dtype=bool)
        if already.size > 0:
            mask[already] = False
        pool = np.arange(n)[mask]
        extra = min(remaining, len(pool))
        if extra > 0:
            chosen.append(rng.choice(pool, extra, replace=False))

    out = np.concatenate(chosen) if chosen else None
    if out is None:
        return None
    if len(np.unique(labels[out])) < 2:
        return None
    return out


def elbow_kmeans_positive_mass(x_train, tau0, k_pos_min=1, k_pos_max=8, use_log1p=False, random_state=0):
    x = np.maximum(np.asarray(x_train, dtype=np.float64), 0.0)
    x_pos = x[x > tau0]
    if x_pos.size == 0:
        raise RuntimeError("No training samples above tau0 for elbow. tau0 too large?")

    x_fit = np.log1p(x_pos) if use_log1p else x_pos
    X = x_fit.reshape(-1, 1)

    rows = []
    for k in range(int(k_pos_min), int(k_pos_max) + 1):
        km = KMeans(n_clusters=int(k), random_state=random_state, n_init=10)
        km.fit(X)
        rows.append({"k_pos": int(k), "inertia": float(km.inertia_)})

    df = pd.DataFrame(rows)

    ks = df["k_pos"].values.astype(float)
    ys = df["inertia"].values.astype(float)
    x1, y1 = ks[0], ys[0]
    x2, y2 = ks[-1], ys[-1]
    denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    if denom < 1e-12:
        k_suggest = int(ks[0])
    else:
        dists = []
        for x0, y0 in zip(ks, ys):
            num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            dists.append(num / denom)
        k_suggest = int(ks[int(np.argmax(dists))])

    return df, k_suggest


def learn_thresholds_kmeans_positive_mass(x_train, pos_k, tau0, use_log1p=False, random_state=0):
    x = np.maximum(np.asarray(x_train, dtype=np.float64), 0.0)
    x_pos = x[x > tau0]
    if x_pos.size == 0:
        raise RuntimeError("No training samples above tau0. tau0 too large?")

    x_fit = np.log1p(x_pos) if use_log1p else x_pos
    X = x_fit.reshape(-1, 1)

    km = KMeans(n_clusters=int(pos_k), random_state=random_state, n_init=10)
    km_labels = km.fit_predict(X)
    inertia = float(km.inertia_)

    centers = km.cluster_centers_.reshape(-1)
    sorted_idx = np.argsort(centers)
    cluster_vals_pos = [x_pos[km_labels == ci] for ci in sorted_idx]

    thresholds_pos = []
    for i in range(pos_k - 1):
        low_v = cluster_vals_pos[i]
        high_v = cluster_vals_pos[i + 1]
        tau = (float(low_v.mean()) + float(high_v.mean())) / 2.0
        thresholds_pos.append(float(tau))

    idx = _stratified_sample_indices(km_labels, sample_n=50000, min_per_cluster=50, seed=42)
    sil = 0.0 if idx is None else float(silhouette_score(X[idx], km_labels[idx]))

    return thresholds_pos, cluster_vals_pos, sil, inertia


def discretize_total_with_tau0(values, tau0, thresholds_pos):
    v = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    out = np.zeros(v.shape, dtype=np.int32)

    mask = v > tau0
    if not np.any(mask):
        return out

    sub = v[mask]
    lab = np.zeros(sub.shape, dtype=np.int32)
    for i, tau in enumerate(sorted(thresholds_pos)):
        lab[sub >= tau] = i + 1
    out[mask] = lab + 1
    return out


def evaluate_test_total(y_true, y_pred, tau0, thresholds_pos, labels_total, min_test_samples=20):
    gt = discretize_total_with_tau0(y_true, tau0, thresholds_pos)
    pr = discretize_total_with_tau0(y_pred, tau0, thresholds_pos)

    total_k = len(labels_total)

    mcc = float(matthews_corrcoef(gt, pr))
    f1_macro = float(f1_score(gt, pr, average="macro", zero_division=0))
    f1_weighted = float(f1_score(gt, pr, average="weighted", zero_division=0))
    f1_per = f1_score(gt, pr, average=None, zero_division=0, labels=list(range(total_k)))

    class_details = []
    evaluable_f1s = []
    for i in range(total_k):
        gt_count = int((gt == i).sum())
        pr_count = int((pr == i).sum())
        f1_i = float(f1_per[i]) if i < len(f1_per) else 0.0
        evaluable = gt_count >= min_test_samples
        if evaluable:
            evaluable_f1s.append(f1_i)

        class_details.append(
            {
                "class": i,
                "label": labels_total[i] if i < len(labels_total) else f"c{i}",
                "gt_count": gt_count,
                "pred_count": pr_count,
                "gt_frac": gt_count / len(gt),
                "f1": f1_i,
                "evaluable": evaluable,
                "note": "" if evaluable else f"only {gt_count} test samples, F1 unreliable",
            }
        )

    f1_evaluable_macro = float(np.mean(evaluable_f1s)) if evaluable_f1s else 0.0
    n_evaluable = len(evaluable_f1s)

    boundary_details = []
    tau_list = [tau0] + list(thresholds_pos)
    for bi, tau in enumerate(tau_list):
        if bi == 0:
            gt_bin = (np.maximum(y_true, 0) > tau).astype(np.int32)
            pr_bin = (np.maximum(y_pred, 0) > tau).astype(np.int32)
        else:
            gt_bin = (np.maximum(y_true, 0) >= tau).astype(np.int32)
            pr_bin = (np.maximum(y_pred, 0) >= tau).astype(np.int32)

        gt_above = int(gt_bin.sum())
        evaluable = gt_above >= min_test_samples and (len(gt_bin) - gt_above) >= min_test_samples

        bmcc = float(matthews_corrcoef(gt_bin, pr_bin))
        bf1 = float(f1_score(gt_bin, pr_bin, zero_division=0))

        left = labels_total[bi]
        right = labels_total[bi + 1]
        boundary_details.append(
            {
                "boundary": f"{left}|{right}",
                "tau": float(tau),
                "binary_mcc": bmcc,
                "binary_f1": bf1,
                "gt_above_count": gt_above,
                "evaluable": evaluable,
                "note": "" if evaluable else f"only {gt_above} above threshold",
            }
        )

    return {
        "mcc": mcc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_evaluable_macro": f1_evaluable_macro,
        "n_evaluable": n_evaluable,
        "class_details": class_details,
        "boundary_details": boundary_details,
    }


def main():
    config_path = "./state_learning/kmeans_state_config.json"
    cfg = load_config(config_path)

    DEVICE_NAME = cfg["DEVICE_NAME"]
    TRAIN_PACKET_CSV = cfg["TRAIN_PACKET_CSV"]
    TEST_TRUE_PRED_CSV = cfg["TEST_TRUE_PRED_CSV"]
    TRAIN_DAYS = cfg.get("TRAIN_DAYS", None)

    BIN_SECONDS = cfg.get("BIN_SECONDS", 1)
    K_TOTAL_LIST = cfg.get("K_TOTAL_LIST", [2, 3, 4, 5, 6])
    USE_LOG1P = cfg.get("USE_LOG1P", False)
    RANDOM_STATE = cfg.get("RANDOM_STATE", 0)
    TAU0_FIXED = cfg.get("TAU0_FIXED", 54.0)

    ELBOW_K_POS_MIN = cfg.get("ELBOW_K_POS_MIN", 1)
    ELBOW_K_POS_MAX = cfg.get("ELBOW_K_POS_MAX", 8)
    MIN_TEST_SAMPLES = cfg.get("MIN_TEST_SAMPLES", 20)

    out_dir = cfg.get("OUT_DIR", f"./state_learning/k_selection_kmeans_{sanitize_filename(DEVICE_NAME)}")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    ts_train = aggregate_packets_to_bins(TRAIN_PACKET_CSV, bin_seconds=BIN_SECONDS, keep_days=TRAIN_DAYS)
    ts_train = complete_index(ts_train, bin_seconds=BIN_SECONDS)

    df_test = pd.read_csv(TEST_TRUE_PRED_CSV)
    true_col = "TRUE" if "TRUE" in df_test.columns else "true"
    y_true = pd.to_numeric(df_test[true_col], errors="coerce").fillna(0.0).values
    y_pred = pd.to_numeric(df_test["pred"], errors="coerce").fillna(0.0).values

    print(f"Train: {len(ts_train):,} points, Test: {len(y_true):,} points")
    print(f"Fixed tau0 (idle|nonidle) = {TAU0_FIXED:.3f}\n")

    df_elbow, kpos_suggest = elbow_kmeans_positive_mass(
        ts_train.values,
        tau0=TAU0_FIXED,
        k_pos_min=ELBOW_K_POS_MIN,
        k_pos_max=ELBOW_K_POS_MAX,
        use_log1p=USE_LOG1P,
        random_state=RANDOM_STATE,
    )

    print("Elbow (positive mass x>tau0) reference:")
    print(df_elbow.to_string(index=False))
    print(f"Suggested k_pos by knee heuristic: {kpos_suggest}")
    print(f"Suggested total classes k_total = idle + k_pos = {1 + kpos_suggest}\n")

    elbow_csv = os.path.join(out_dir, f"{sanitize_filename(DEVICE_NAME)}_elbow_pos_tau0_{int(TAU0_FIXED)}.csv")
    df_elbow.to_csv(elbow_csv, index=False)

    elbow_png = os.path.join(out_dir, f"{sanitize_filename(DEVICE_NAME)}_elbow_pos_tau0_{int(TAU0_FIXED)}.png")
    plt.figure()
    plt.plot(df_elbow["k_pos"].values, df_elbow["inertia"].values, marker="o")
    plt.xlabel("k_pos (clusters on x>tau0 only)")
    plt.ylabel("inertia (SSE)")
    plt.title(f"Elbow on positive mass (tau0={TAU0_FIXED:.0f}): {DEVICE_NAME}")
    plt.savefig(elbow_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved elbow: {elbow_csv}")
    print(f"Saved elbow figure: {elbow_png}\n")

    summary_rows = []

    print(f"\n{'='*90}")
    print(f"KMeans on x>tau0, tau0 fixed at {TAU0_FIXED:.0f}, thresholds by midpoint")
    print(f"{'='*90}")

    for k_total in K_TOTAL_LIST:
        if k_total < 2:
            continue

        pos_k = int(k_total) - 1
        labels_total = SEMANTIC_LABELS_TOTAL.get(
            k_total,
            ["idle"] + [f"c{i}" for i in range(1, k_total)],
        )

        thresholds_pos, cluster_vals_pos, sil_pos, inertia_pos = learn_thresholds_kmeans_positive_mass(
            ts_train.values,
            pos_k=pos_k,
            tau0=TAU0_FIXED,
            use_log1p=USE_LOG1P,
            random_state=RANDOM_STATE,
        )

        validity_pos = evaluate_cluster_validity(cluster_vals_pos)

        print(f"\n{'-'*90}")
        print(f"k_total={k_total}  pos_k={pos_k}  Sil_pos={sil_pos:.4f}  Inertia_pos={inertia_pos:.2f}")
        print(f"tau0={TAU0_FIXED:.1f}  thresholds_pos={[f'{t:.1f}' for t in thresholds_pos]}")
        print(f"{'-'*90}")

        print("\n[TRAIN] Positive components validity (x>tau0 only):")
        print(f"{'Rank':<5} {'Label':<12} {'Count':>10} {'Mean':>10} {'Width(p5-p95)':>14} {'Gap→neighbor':>14} {'Sep.Ratio':>10} {'Valid?':>8}")
        n_valid_pos = 0
        for v in validity_pos:
            rank = v["cluster_rank"]
            label = labels_total[rank + 1] if (rank + 1) < len(labels_total) else f"pos{rank}"
            valid_str = "Y" if v["is_separable"] else "N"
            if v["is_separable"]:
                n_valid_pos += 1
            print(
                f"{rank:<5} {label:<12} {v['count']:>10,} {v['mean']:>10.1f} "
                f"{v['width_p5_p95']:>14.1f} {v['min_gap_to_neighbor']:>14.1f} "
                f"{v['separation_ratio']:>10.2f} {valid_str:>8}"
            )

        test_result = evaluate_test_total(
            y_true,
            y_pred,
            tau0=TAU0_FIXED,
            thresholds_pos=thresholds_pos,
            labels_total=labels_total,
            min_test_samples=MIN_TEST_SAMPLES,
        )

        print("\n[TEST] evaluation (GT and pred use same tau0+thresholds):")
        print(
            f"MCC={test_result['mcc']:.6f}  "
            f"F1_macro={test_result['f1_macro']:.6f}  "
            f"F1_weighted={test_result['f1_weighted']:.6f}  "
            f"F1_evaluable_macro={test_result['f1_evaluable_macro']:.6f} "
            f"({test_result['n_evaluable']}/{k_total} classes evaluable)"
        )

        print(f"\n{'Class':<12} {'GT#':>10} {'Pred#':>10} {'GT%':>10} {'F1':>10} {'Status'}")
        for cd in test_result["class_details"]:
            status = "eval" if cd["evaluable"] else cd["note"]
            print(
                f"{cd['label']:<12} {cd['gt_count']:>10,} {cd['pred_count']:>10,} "
                f"{cd['gt_frac']:>10.4f} {cd['f1']:>10.4f} {status}"
            )

        print("\nPer-boundary (idle|first-positive then positives):")
        for bd in test_result["boundary_details"]:
            status = "eval" if bd["evaluable"] else bd["note"]
            print(
                f"{bd['boundary']:<20} tau={bd['tau']:>10.1f}  "
                f"MCC={bd['binary_mcc']:.4f}  F1={bd['binary_f1']:.4f}  {status}"
            )

        summary_rows.append(
            {
                "method": "midpoint",
                "k_total": k_total,
                "pos_k": pos_k,
                "tau0": TAU0_FIXED,
                "sil_pos": sil_pos,
                "inertia_pos": inertia_pos,
                "n_valid_pos": n_valid_pos,
                "thresholds_pos": str([round(t, 1) for t in thresholds_pos]),
                "mcc": test_result["mcc"],
                "f1_macro": test_result["f1_macro"],
                "f1_weighted": test_result["f1_weighted"],
                "f1_evaluable_macro": test_result["f1_evaluable_macro"],
                "min_sep_pos": min(v["separation_ratio"] for v in validity_pos) if validity_pos else 0.0,
                "mean_sep_pos": float(np.mean([v["separation_ratio"] for v in validity_pos])) if validity_pos else 0.0,
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    print(f"\n\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    print(
        f"{'Method':<16} {'kTot':>4} {'posK':>4} {'Sil':>7} {'Iner':>12} {'ValidPos':>9} "
        f"{'MCC':>8} {'F1_wt':>8} {'F1_eval':>8} {'MinSepPos':>10}"
    )
    print("-" * 90)
    for _, r in df_summary.iterrows():
        print(
            f"{r['method']:<16} {int(r['k_total']):>4} {int(r['pos_k']):>4} {r['sil_pos']:>7.3f} "
            f"{r['inertia_pos']:>12.1f} {int(r['n_valid_pos']):>5}/{int(r['pos_k']):<3} "
            f"{r['mcc']:>8.4f} {r['f1_weighted']:>8.4f} {r['f1_evaluable_macro']:>8.4f} {r['min_sep_pos']:>10.2f}"
        )

    summary_csv = os.path.join(out_dir, f"{sanitize_filename(DEVICE_NAME)}_kmeans_tau0_{int(TAU0_FIXED)}_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()