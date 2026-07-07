from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "visualization"
LONG_CSV = OUTPUT_DIR / "纳入文件_0-20秒原始波形长表.csv"
FEATURE_CSV = OUTPUT_DIR / "纳入文件_可视化特征表.csv"
RESAMPLED_CSV = OUTPUT_DIR / "纳入文件_重采样基线校正波形.csv"
SINGLE_DIR = OUTPUT_DIR / "单细胞波形图_标记峰谷"
OVERVIEW_DIR = OUTPUT_DIR / "总览图"

STIM_ON = 5.0
STIM_OFF = 10.0
COMMON_END = 20.0
RESAMPLE_DT = 0.05
SMOOTH_SECONDS = 0.25
PEAK_MIN_DISTANCE_SECONDS = 0.30
PEAK_ABS_THRESHOLD = 0.003
PEAK_PROMINENCE_MIN = 0.0015
BASELINE_STD_MULTIPLIER = 3.0


def segment_mask(time, start, end):
    return (time >= start) & (time < end)


def safe_auc(time, y):
    if len(time) < 2:
        return np.nan
    return float(np.trapezoid(y, time))


def safe_slope(time, y):
    if len(time) < 2:
        return np.nan
    x = time - time[0]
    if np.nanstd(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def smooth_signal(y):
    window = max(5, int(round(SMOOTH_SECONDS / RESAMPLE_DT)))
    if window % 2 == 0:
        window += 1
    if window >= len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    if window < 5:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=2)


def segment_stats(prefix, time, y, start, end):
    mask = segment_mask(time, start, end)
    values = y[mask]
    t = time[mask]
    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_auc": np.nan,
            f"{prefix}_slope": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_max": float(np.nanmax(values)),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_auc": safe_auc(t, values),
        f"{prefix}_slope": safe_slope(t, values),
    }


def count_events(times, amps, start, end, use_min=False):
    times = np.asarray(times)
    amps = np.asarray(amps)
    mask = (times >= start) & (times < end)
    t = times[mask]
    a = amps[mask]
    if len(a) == 0:
        return 0, 0.0, np.nan, np.nan
    idx = int(np.argmin(a) if use_min else np.argmax(a))
    return int(len(a)), float(a[idx]), float(t[idx]), float(t[0])


def extract_one(group):
    group = group.sort_values("time")
    time_raw = group["time"].to_numpy(dtype=float)
    f_raw = group["f_f0"].to_numpy(dtype=float)
    common_time = np.arange(0, COMMON_END + RESAMPLE_DT / 2, RESAMPLE_DT)
    f_interp = np.interp(common_time, time_raw, f_raw)
    f_smooth = smooth_signal(f_interp)

    pre = segment_mask(common_time, 0, STIM_ON)
    baseline_mean = float(np.nanmean(f_smooth[pre]))
    baseline_std = float(np.nanstd(f_smooth[pre]))
    y = f_smooth - baseline_mean

    threshold = max(PEAK_ABS_THRESHOLD, BASELINE_STD_MULTIPLIER * baseline_std)
    prominence = max(PEAK_PROMINENCE_MIN, 1.5 * baseline_std)
    distance = max(1, int(round(PEAK_MIN_DISTANCE_SECONDS / RESAMPLE_DT)))

    peak_idx, _ = find_peaks(y, height=threshold, prominence=prominence, distance=distance)
    valley_idx, _ = find_peaks(-y, height=threshold, prominence=prominence, distance=distance)

    peak_times = common_time[peak_idx]
    peak_amps = y[peak_idx]
    valley_times = common_time[valley_idx]
    valley_amps = y[valley_idx]

    feature = {
        "file_id": group["file_id"].iloc[0],
        "filename": group["filename"].iloc[0],
        "source_relative_path": group["source_relative_path"].iloc[0],
        "baseline_mean_0_5s": baseline_mean,
        "baseline_std_0_5s": baseline_std,
        "peak_threshold_used": float(threshold),
        "peak_prominence_used": float(prominence),
        "total_peak_count_0_20s": int(len(peak_idx)),
        "total_valley_count_0_20s": int(len(valley_idx)),
        "max_peak_amp_0_20s": float(np.max(peak_amps)) if len(peak_amps) else 0.0,
        "max_peak_time_0_20s": float(peak_times[np.argmax(peak_amps)]) if len(peak_amps) else np.nan,
        "max_valley_amp_0_20s": float(np.min(valley_amps)) if len(valley_amps) else 0.0,
        "max_valley_time_0_20s": float(valley_times[np.argmin(valley_amps)]) if len(valley_amps) else np.nan,
        "first_peak_time_0_20s": float(peak_times[0]) if len(peak_times) else np.nan,
        "first_valley_time_0_20s": float(valley_times[0]) if len(valley_times) else np.nan,
        "response_range_0_20s": float(np.nanmax(y) - np.nanmin(y)),
        "response_std_0_20s": float(np.nanstd(y)),
    }

    for prefix, start, end in [
        ("pre_0_5s", 0, STIM_ON),
        ("stim_5_10s", STIM_ON, STIM_OFF),
        ("post_10_20s", STIM_OFF, COMMON_END + RESAMPLE_DT),
    ]:
        feature.update(segment_stats(prefix, common_time, y, start, end))
        pc, pa, pt, pfirst = count_events(peak_times, peak_amps, start, end)
        vc, va, vt, vfirst = count_events(valley_times, valley_amps, start, end, use_min=True)
        feature[f"{prefix}_peak_count"] = pc
        feature[f"{prefix}_max_peak_amp"] = pa
        feature[f"{prefix}_max_peak_time"] = pt
        feature[f"{prefix}_first_peak_time"] = pfirst
        feature[f"{prefix}_valley_count"] = vc
        feature[f"{prefix}_max_valley_amp"] = va
        feature[f"{prefix}_max_valley_time"] = vt
        feature[f"{prefix}_first_valley_time"] = vfirst

    feature["has_pre_peak"] = int(feature["pre_0_5s_peak_count"] > 0)
    feature["has_pre_valley"] = int(feature["pre_0_5s_valley_count"] > 0)
    feature["has_stim_peak"] = int(feature["stim_5_10s_peak_count"] > 0)
    feature["has_stim_valley"] = int(feature["stim_5_10s_valley_count"] > 0)
    feature["has_post_peak"] = int(feature["post_10_20s_peak_count"] > 0)
    feature["has_post_valley"] = int(feature["post_10_20s_valley_count"] > 0)
    feature["stim_mean_above_baseline"] = int(feature["stim_5_10s_mean"] > 0)
    feature["post_mean_above_baseline"] = int(feature["post_10_20s_mean"] > 0)
    feature["post_max_above_baseline"] = int(feature["post_10_20s_max"] > 0)
    feature["post_min_below_baseline"] = int(feature["post_10_20s_min"] < 0)
    feature["on_peak_amp"] = feature["stim_5_10s_max_peak_amp"]
    feature["on_latency_s"] = (
        feature["stim_5_10s_first_peak_time"] - STIM_ON
        if np.isfinite(feature["stim_5_10s_first_peak_time"])
        else np.nan
    )
    feature["off_peak_amp"] = feature["post_10_20s_max_peak_amp"]
    feature["off_latency_s"] = (
        feature["post_10_20s_first_peak_time"] - STIM_OFF
        if np.isfinite(feature["post_10_20s_first_peak_time"])
        else np.nan
    )
    feature["stim_mean_delta_vs_pre"] = feature["stim_5_10s_mean"] - feature["pre_0_5s_mean"]
    feature["post_mean_delta_vs_pre"] = feature["post_10_20s_mean"] - feature["pre_0_5s_mean"]
    feature["post_recovery_slope"] = feature["post_10_20s_slope"]

    waveform = pd.DataFrame(
        {
            "file_id": feature["file_id"],
            "filename": feature["filename"],
            "source_relative_path": feature["source_relative_path"],
            "time": common_time,
            "f_f0_smooth": f_smooth,
            "baseline_corrected": y,
            "is_peak": 0,
            "is_valley": 0,
        }
    )
    waveform.loc[peak_idx, "is_peak"] = 1
    waveform.loc[valley_idx, "is_valley"] = 1
    return feature, waveform


def plot_single(waveform, feature):
    file_id = feature["file_id"]
    plt.figure(figsize=(9, 4.8))
    plt.plot(waveform["time"], waveform["baseline_corrected"], color="#1f77b4", lw=1.6, label="Smoothed, baseline corrected")
    peaks = waveform[waveform["is_peak"] == 1]
    valleys = waveform[waveform["is_valley"] == 1]
    if len(peaks):
        plt.scatter(peaks["time"], peaks["baseline_corrected"], s=30, color="#d62728", label="Peaks", zorder=3)
    if len(valleys):
        plt.scatter(valleys["time"], valleys["baseline_corrected"], s=30, color="#2ca02c", label="Valleys", zorder=3)
    plt.axvspan(STIM_ON, STIM_OFF, color="#f2c14e", alpha=0.25, label="Stim 5-10s")
    plt.axhline(0, color="black", lw=0.8, alpha=0.65)
    plt.title(f"{file_id} | peaks={feature['total_peak_count_0_20s']}, valleys={feature['total_valley_count_0_20s']}")
    plt.xlabel("Time (s)")
    plt.ylabel("F/F0 minus baseline")
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(SINGLE_DIR / f"{file_id}.png", dpi=180)
    plt.close()


def plot_overview(waveforms, features):
    pivot = waveforms.pivot(index="file_id", columns="time", values="baseline_corrected")
    time = pivot.columns.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(11, 6))
    for row in values:
        plt.plot(time, row, color="#5d7fa3", alpha=0.12, lw=0.8)
    mean = np.nanmean(values, axis=0)
    plt.plot(time, mean, color="#d62728", lw=2.6, label=f"Mean (n={len(values)})")
    plt.axvspan(STIM_ON, STIM_OFF, color="#f2c14e", alpha=0.25, label="Stim 5-10s")
    plt.axhline(0, color="black", lw=0.8, alpha=0.65)
    plt.xlabel("Time (s)")
    plt.ylabel("F/F0 minus baseline")
    plt.title("All included waveforms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OVERVIEW_DIR / "全部波形叠加_含平均.png", dpi=220)
    plt.close()

    sem = np.nanstd(values, axis=0) / np.sqrt(values.shape[0])
    plt.figure(figsize=(10, 5))
    plt.plot(time, mean, color="#1f77b4", lw=2.6, label="Mean")
    plt.fill_between(time, mean - sem, mean + sem, color="#1f77b4", alpha=0.22, label="SEM")
    plt.axvspan(STIM_ON, STIM_OFF, color="#f2c14e", alpha=0.25, label="Stim 5-10s")
    plt.axhline(0, color="black", lw=0.8, alpha=0.65)
    plt.xlabel("Time (s)")
    plt.ylabel("F/F0 minus baseline")
    plt.title("Mean waveform with SEM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OVERVIEW_DIR / "平均波形_SEM.png", dpi=220)
    plt.close()

    ordered_ids = features.sort_values("stim_mean_delta_vs_pre", ascending=False)["file_id"].tolist()
    heat = pivot.loc[ordered_ids]
    plt.figure(figsize=(10, 9))
    plt.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-0.08, vmax=0.08, extent=[time.min(), time.max(), len(heat), 0])
    plt.axvspan(STIM_ON, STIM_OFF, color="#f2c14e", alpha=0.22)
    plt.colorbar(label="F/F0 minus baseline")
    plt.xlabel("Time (s)")
    plt.ylabel("Cells sorted by stim mean")
    plt.title("Waveform heatmap")
    plt.tight_layout()
    plt.savefig(OVERVIEW_DIR / "波形热图_按刺激期均值排序.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    axes[0].hist(features["total_peak_count_0_20s"], bins=range(int(features["total_peak_count_0_20s"].max()) + 2), color="#4c78a8", alpha=0.85)
    axes[0].set_title("Peak count, 0-20s")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Cells")

    axes[1].hist(features["total_valley_count_0_20s"], bins=range(int(features["total_valley_count_0_20s"].max()) + 2), color="#59a14f", alpha=0.85)
    axes[1].set_title("Valley count, 0-20s")
    axes[1].set_xlabel("Count")

    axes[2].hist(features["on_peak_amp"], bins=25, color="#e15759", alpha=0.85)
    axes[2].set_title("ON peak amplitude")
    axes[2].set_xlabel("F/F0 minus baseline")

    axes[3].hist(features["post_mean_delta_vs_pre"], bins=25, color="#f28e2b", alpha=0.85)
    axes[3].set_title("Post mean delta vs baseline")
    axes[3].set_xlabel("F/F0 minus baseline")
    plt.tight_layout()
    plt.savefig(OVERVIEW_DIR / "特征分布_峰谷和响应幅度.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(features["on_peak_amp"], features["post_mean_delta_vs_pre"], s=38, alpha=0.78, c=features["total_peak_count_0_20s"], cmap="viridis")
    plt.axhline(0, color="black", lw=0.8, alpha=0.45)
    plt.axvline(0, color="black", lw=0.8, alpha=0.45)
    plt.colorbar(label="Peak count 0-20s")
    plt.xlabel("ON peak amplitude")
    plt.ylabel("Post mean delta vs baseline")
    plt.title("ON response vs post-stimulus level")
    plt.tight_layout()
    plt.savefig(OVERVIEW_DIR / "散点_ON幅度_vs_撤光后均值.png", dpi=220)
    plt.close()


def main():
    SINGLE_DIR.mkdir(parents=True, exist_ok=True)
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(LONG_CSV)
    feature_rows = []
    waveform_rows = []

    for i, (file_id, group) in enumerate(raw.groupby("file_id", sort=True), start=1):
        feature, waveform = extract_one(group)
        feature_rows.append(feature)
        waveform_rows.append(waveform)
        plot_single(waveform, feature)
        if i % 25 == 0 or i == raw["file_id"].nunique():
            print(f"已生成 {i}/{raw['file_id'].nunique()} 个单细胞波形图")

    features = pd.DataFrame(feature_rows)
    waveforms = pd.concat(waveform_rows, ignore_index=True)
    features.to_csv(FEATURE_CSV, index=False, encoding="utf-8-sig")
    waveforms.to_csv(RESAMPLED_CSV, index=False, encoding="utf-8-sig")
    plot_overview(waveforms, features)

    print(f"可视化结果: {OUTPUT_DIR}")
    print(f"单细胞波形图: {SINGLE_DIR}")
    print(f"总览图: {OVERVIEW_DIR}")
    print(f"特征表: {FEATURE_CSV}")


if __name__ == "__main__":
    main()
