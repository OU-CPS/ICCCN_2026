from pathlib import Path
import numpy as np
import pandas as pd


def aggregate_device_packets(device_csv: str, bin_seconds: int = 60, keep_days=None) -> pd.Series:
    df = pd.read_csv(device_csv)

    if keep_days is not None:
        df = df[df["day"].isin(keep_days)]

    df["TIME"] = pd.to_numeric(df["TIME"], errors="coerce")
    df["Size"] = pd.to_numeric(df["Size"], errors="coerce").fillna(0.0)

    df.loc[df["Size"] < 0, "Size"] = 0.0
    df = df.dropna(subset=["TIME"])

    t = df["TIME"].astype(np.int64)
    df["tbin"] = (t // bin_seconds) * bin_seconds

    ts = df.groupby("tbin")["Size"].sum().sort_index()

    print(f"[aggregate_device_packets] {device_csv}")
    print(ts.head(5))
    print(ts.tail(5))

    return ts


def build_dev_series_selected(
    device_dir: str,
    selected_devices,
    bin_seconds: int = 60,
    suffix: str = "downlink.csv",
    keep_days=None
) -> pd.DataFrame:
    series_list = []

    for dev in selected_devices:
        fp = Path(device_dir) / f"{dev} {suffix}"
        if not fp.exists():
            raise FileNotFoundError(f"file not found: {fp}")

        ts = aggregate_device_packets(str(fp), bin_seconds=bin_seconds, keep_days=keep_days)
        ts.name = dev
        series_list.append(ts)

    dev_df = pd.concat(series_list, axis=1).fillna(0.0).sort_index()

    t0 = int(dev_df.index.min())
    t1 = int(dev_df.index.max())
    full_index = np.arange(t0, t1 + bin_seconds, bin_seconds, dtype=np.int64)

    dev_df = dev_df.reindex(full_index).fillna(0.0).sort_index()
    return dev_df


def align_up_down(dev_up: pd.DataFrame, dev_down: pd.DataFrame):
    common_index = dev_up.index.intersection(dev_down.index)
    dev_up = dev_up.reindex(common_index).fillna(0.0).sort_index()
    dev_down = dev_down.reindex(common_index).fillna(0.0).sort_index()
    return dev_up, dev_down