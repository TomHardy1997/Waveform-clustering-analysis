from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = PROJECT_ROOT / "results" / "visualization"
LONG_CSV = VIS_DIR / "纳入文件_0-20秒原始波形长表.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "preprocessing"
SINGLE_DATA_DIR = OUTPUT_DIR / "单文件CSV数据"
PLOT_DIR = OUTPUT_DIR / "单文件预处理波形图"
OVERVIEW_DIR = OUTPUT_DIR / "总览图"

PREPROCESS_CSV = SINGLE_DATA_DIR / "全部文件_预处理后数据.csv"
PEAK_CSV = SINGLE_DATA_DIR / "全部文件_峰值特征.csv"
VALLEY_CSV = SINGLE_DATA_DIR / "全部文件_波谷特征.csv"
STAGE_CSV = SINGLE_DATA_DIR / "全部文件_分阶段特征.csv"
SUMMARY_CSV = OUTPUT_DIR / "全部文件_整体统计和核心特征.csv"
CORE_FEATURE_CSV = OUTPUT_DIR / "用于聚类的核心特征表.csv"

STIM_ON = 5.0
STIM_OFF = 10.0
COMMON_END = 20.0

BASELINE_RATIO = 0.25
SAVGOL_WINDOW = 51
SAVGOL_POLYORDER = 3
PEAK_PROMINENCE = 0.002
PEAK_HEIGHT = 0.003
PEAK_DISTANCE_SECONDS = 0.30
VALLEY_PROMINENCE = 0.002
VALLEY_DEPTH = 0.003


def ensure_odd_window(window, n):
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window < 5:
        return None
    return window


def segment_mask(time, start, end):
    return (time >= start) & (time < end)


def safe_auc(time, y):
    if len(time) < 2:
        return np.nan
    return float(np.trapezoid(y, time))


def safe_slope(time, y):
    if len(time) < 2 or np.nanstd(time) == 0:
        return np.nan
    return float(np.polyfit(time - time[0], y, 1)[0])


def segment_stats(prefix, time, y):
    ranges = {
        "pre_0_5s": (0, STIM_ON),
        "stim_5_10s": (STIM_ON, STIM_OFF),
        "post_10_20s": (STIM_OFF, COMMON_END + 1e-9),
    }
    start, end = ranges[prefix]
    mask = segment_mask(time, start, end)
    vals = y[mask]
    t = time[mask]
    if len(vals) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_auc": np.nan,
            f"{prefix}_slope": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.nanmean(vals)),
        f"{prefix}_max": float(np.nanmax(vals)),
        f"{prefix}_min": float(np.nanmin(vals)),
        f"{prefix}_std": float(np.nanstd(vals)),
        f"{prefix}_auc": safe_auc(t, vals),
        f"{prefix}_slope": safe_slope(t, vals),
    }


def count_events(times, amps, start, end, strongest="max"):
    times = np.asarray(times)
    amps = np.asarray(amps)
    mask = (times >= start) & (times < end)
    t = times[mask]
    a = amps[mask]
    if len(a) == 0:
        return 0, 0.0, np.nan, np.nan
    idx = int(np.argmin(a) if strongest == "min" else np.argmax(a))
    return int(len(a)), float(a[idx]), float(t[idx]), float(t[0])


def classify_response(row):
    on = row.get("ON反应幅度", row.get("on_peak_amp", 0))
    off = row.get("OFF反应幅度", row.get("off_peak_amp", 0))
    stim_delta = row.get("刺激期相对刺激前变化量", row.get("stim_mean_delta_vs_pre", 0))
    if on > PEAK_HEIGHT and off > PEAK_HEIGHT:
        return "ON-OFF型"
    if on > PEAK_HEIGHT:
        return "ON型"
    if off > PEAK_HEIGHT:
        return "OFF型"
    if stim_delta < -PEAK_HEIGHT:
        return "抑制型"
    return "弱反应或无反应型"


def preprocess_one(group):
    group = group.sort_values("time")
    file_id = group["file_id"].iloc[0]
    filename = group["filename"].iloc[0]
    source_relative_path = group["source_relative_path"].iloc[0]

    time = group["time"].to_numpy(dtype=float)
    f_raw = group["f_f0"].to_numpy(dtype=float)
    dff_raw = group["dff"].to_numpy(dtype=float)
    if np.isnan(dff_raw).all():
        dff_raw = f_raw - 1

    dt = float(np.nanmedian(np.diff(time)))
    sample_rate = 1 / dt

    mean_f = float(np.nanmean(f_raw))
    std_f = float(np.nanstd(f_raw))
    f_clip = np.clip(f_raw, mean_f - 3 * std_f, mean_f + 3 * std_f)

    baseline_length = max(3, int(len(f_clip) * BASELINE_RATIO))
    baseline_value = float(np.nanmean(f_clip[:baseline_length]))
    f_baseline = f_clip / baseline_value

    window = ensure_odd_window(SAVGOL_WINDOW, len(f_baseline))
    if window is None:
        f_smooth = f_baseline.copy()
        dff_smooth = dff_raw.copy()
    else:
        f_smooth = savgol_filter(f_baseline, window_length=window, polyorder=SAVGOL_POLYORDER)
        dff_smooth = savgol_filter(dff_raw, window_length=window, polyorder=SAVGOL_POLYORDER)

    baseline_corrected = f_smooth - 1
    peak_distance = max(1, int(sample_rate * PEAK_DISTANCE_SECONDS))

    peaks, peak_props = find_peaks(
        baseline_corrected,
        height=PEAK_HEIGHT,
        prominence=PEAK_PROMINENCE,
        distance=peak_distance,
    )
    valleys, valley_props = find_peaks(
        -baseline_corrected,
        height=VALLEY_DEPTH,
        prominence=VALLEY_PROMINENCE,
        distance=peak_distance,
    )

    preprocess = pd.DataFrame(
        {
            "file_id": file_id,
            "文件名": filename,
            "时间(秒)": time,
            "原始F/F0": f_raw,
            "异常值截断后F/F0": f_clip,
            "基线校正后F/F0": f_baseline,
            "平滑去噪后F/F0": f_smooth,
            "基线校正后ΔF/F0": baseline_corrected,
            "原始dF/F0": dff_raw,
            "平滑去噪后dF/F0": dff_smooth,
        }
    )

    peak_rows = []
    if len(peaks):
        widths, width_heights, left_ips, right_ips = peak_widths(baseline_corrected, peaks, rel_height=0.5)
        for i, peak_idx in enumerate(peaks):
            amp = float(baseline_corrected[peak_idx])
            rise_threshold = 0.1 * amp
            left_candidates = np.where(baseline_corrected[:peak_idx] <= rise_threshold)[0]
            rise_start = int(left_candidates[-1]) if len(left_candidates) else 0
            right_candidates = np.where(baseline_corrected[peak_idx:] <= rise_threshold)[0]
            fall_end = int(right_candidates[0] + peak_idx) if len(right_candidates) else len(baseline_corrected) - 1
            rise_time = float(time[peak_idx] - time[rise_start])
            fall_time = float(time[fall_end] - time[peak_idx])
            peak_rows.append(
                {
                    "file_id": file_id,
                    "文件名": filename,
                    "峰编号": i + 1,
                    "峰值时间(秒)": round(float(time[peak_idx]), 4),
                    "峰值幅度(ΔF/F0)": round(amp, 6),
                    "半高宽(秒)": round(float(widths[i] / sample_rate), 4),
                    "半高宽左边界(秒)": round(float(time[0] + left_ips[i] / sample_rate), 4),
                    "半高宽右边界(秒)": round(float(time[0] + right_ips[i] / sample_rate), 4),
                    "10%-90%上升时间(秒)": round(rise_time, 4),
                    "平均上升速率(ΔF/F0/秒)": round(float(amp / rise_time), 6) if rise_time > 0 else np.nan,
                    "90%-10%下降时间(秒)": round(fall_time, 4),
                    "平均下降速率(ΔF/F0/秒)": round(float(amp / fall_time), 6) if fall_time > 0 else np.nan,
                }
            )
    peak_df = pd.DataFrame(peak_rows)

    valley_rows = []
    if len(valleys):
        widths, width_heights, left_ips, right_ips = peak_widths(-baseline_corrected, valleys, rel_height=0.5)
        for i, valley_idx in enumerate(valleys):
            amp = float(baseline_corrected[valley_idx])
            valley_rows.append(
                {
                    "file_id": file_id,
                    "文件名": filename,
                    "谷编号": i + 1,
                    "波谷时间(秒)": round(float(time[valley_idx]), 4),
                    "波谷幅度(ΔF/F0)": round(amp, 6),
                    "波谷半高宽(秒)": round(float(widths[i] / sample_rate), 4),
                    "半高宽左边界(秒)": round(float(time[0] + left_ips[i] / sample_rate), 4),
                    "半高宽右边界(秒)": round(float(time[0] + right_ips[i] / sample_rate), 4),
                }
            )
    valley_df = pd.DataFrame(valley_rows)

    peak_times = np.array([r["峰值时间(秒)"] for r in peak_rows], dtype=float) if peak_rows else np.array([])
    peak_amps = np.array([r["峰值幅度(ΔF/F0)"] for r in peak_rows], dtype=float) if peak_rows else np.array([])
    valley_times = np.array([r["波谷时间(秒)"] for r in valley_rows], dtype=float) if valley_rows else np.array([])
    valley_amps = np.array([r["波谷幅度(ΔF/F0)"] for r in valley_rows], dtype=float) if valley_rows else np.array([])

    stage = {
        "file_id": file_id,
        "文件名": filename,
    }
    for prefix in ["pre_0_5s", "stim_5_10s", "post_10_20s"]:
        stage.update(segment_stats(prefix, time, baseline_corrected))

    for label, start, end in [
        ("pre_0_5s", 0, STIM_ON),
        ("stim_5_10s", STIM_ON, STIM_OFF),
        ("post_10_20s", STIM_OFF, COMMON_END + 1e-9),
    ]:
        pc, pa, pt, pfirst = count_events(peak_times, peak_amps, start, end, strongest="max")
        vc, va, vt, vfirst = count_events(valley_times, valley_amps, start, end, strongest="min")
        stage[f"{label}_peak_count"] = pc
        stage[f"{label}_max_peak_amp"] = pa
        stage[f"{label}_max_peak_time"] = pt
        stage[f"{label}_first_peak_time"] = pfirst
        stage[f"{label}_valley_count"] = vc
        stage[f"{label}_max_valley_amp"] = va
        stage[f"{label}_max_valley_time"] = vt
        stage[f"{label}_first_valley_time"] = vfirst

    stage["刺激前是否有波峰"] = int(stage["pre_0_5s_peak_count"] > 0)
    stage["刺激前是否有波谷"] = int(stage["pre_0_5s_valley_count"] > 0)
    stage["刺激期是否有波峰"] = int(stage["stim_5_10s_peak_count"] > 0)
    stage["刺激期是否有波谷"] = int(stage["stim_5_10s_valley_count"] > 0)
    stage["撤光后是否有波峰"] = int(stage["post_10_20s_peak_count"] > 0)
    stage["撤光后是否有波谷"] = int(stage["post_10_20s_valley_count"] > 0)
    stage["撤光后均值是否高于基线"] = int(stage["post_10_20s_mean"] > 0)
    stage["撤光后最高点是否高于基线"] = int(stage["post_10_20s_max"] > 0)
    stage["撤光后最低点是否低于基线"] = int(stage["post_10_20s_min"] < 0)

    peak_count = len(peak_df)
    valley_count = len(valley_df)
    duration = float(time[-1] - time[0])
    peak_interval = np.diff(peak_times) if len(peak_times) > 1 else np.array([])
    max_peak_amp = float(peak_amps.max()) if len(peak_amps) else np.nan
    max_peak_time = float(peak_times[np.argmax(peak_amps)]) if len(peak_amps) else np.nan
    min_valley_amp = float(valley_amps.min()) if len(valley_amps) else np.nan
    min_valley_time = float(valley_times[np.argmin(valley_amps)]) if len(valley_amps) else np.nan

    summary = {
        "file_id": file_id,
        "文件名": filename,
        "来源路径": source_relative_path,
        "总时长(秒)": round(duration, 4),
        "采样频率(Hz)": round(sample_rate, 4),
        "总数据点数": len(time),
        "F/F0基线值": round(baseline_value, 6),
        "F/F0全时段均值": round(float(np.nanmean(f_smooth)), 6),
        "F/F0全时段标准差": round(float(np.nanstd(f_smooth)), 6),
        "F/F0最大值": round(float(np.nanmax(f_smooth)), 6),
        "F/F0最小值": round(float(np.nanmin(f_smooth)), 6),
        "dF/F0全时段均值": round(float(np.nanmean(dff_smooth)), 6),
        "dF/F0全时段标准差": round(float(np.nanstd(dff_smooth)), 6),
        "总波峰数量": peak_count,
        "总波谷数量": valley_count,
        "最大波峰幅度": round(max_peak_amp, 6) if np.isfinite(max_peak_amp) else np.nan,
        "最大波峰出现时间": round(max_peak_time, 4) if np.isfinite(max_peak_time) else np.nan,
        "最大波谷幅度": round(min_valley_amp, 6) if np.isfinite(min_valley_amp) else np.nan,
        "最大波谷出现时间": round(min_valley_time, 4) if np.isfinite(min_valley_time) else np.nan,
        "平均波峰幅度": round(float(np.nanmean(peak_amps)), 6) if len(peak_amps) else np.nan,
        "平均波谷幅度": round(float(np.nanmean(valley_amps)), 6) if len(valley_amps) else np.nan,
        "波峰平均宽度": round(float(peak_df["半高宽(秒)"].mean()), 4) if peak_count else np.nan,
        "波峰平均上升时间": round(float(peak_df["10%-90%上升时间(秒)"].mean()), 4) if peak_count else np.nan,
        "波峰平均下降时间": round(float(peak_df["90%-10%下降时间(秒)"].mean()), 4) if peak_count else np.nan,
        "波峰发放频率(个/秒)": round(float(peak_count / duration), 6) if duration > 0 else np.nan,
        "第一个波峰出现时间": round(float(peak_times[0]), 4) if len(peak_times) else np.nan,
        "最后一个波峰出现时间": round(float(peak_times[-1]), 4) if len(peak_times) else np.nan,
        "第一个波谷出现时间": round(float(valley_times[0]), 4) if len(valley_times) else np.nan,
        "峰间间隔平均值": round(float(np.nanmean(peak_interval)), 4) if len(peak_interval) else np.nan,
        "峰间间隔变异性": round(float(np.nanstd(peak_interval)), 4) if len(peak_interval) else np.nan,
        "ON反应幅度": stage["stim_5_10s_max_peak_amp"],
        "ON反应潜伏期": stage["stim_5_10s_first_peak_time"] - STIM_ON if np.isfinite(stage["stim_5_10s_first_peak_time"]) else np.nan,
        "OFF反应幅度": stage["post_10_20s_max_peak_amp"],
        "OFF反应潜伏期": stage["post_10_20s_first_peak_time"] - STIM_OFF if np.isfinite(stage["post_10_20s_first_peak_time"]) else np.nan,
        "ON/OFF比值": stage["stim_5_10s_max_peak_amp"] / (stage["post_10_20s_max_peak_amp"] + 1e-9),
        "刺激期相对刺激前变化量": stage["stim_5_10s_mean"] - stage["pre_0_5s_mean"],
        "撤光后相对刺激前变化量": stage["post_10_20s_mean"] - stage["pre_0_5s_mean"],
        "曲线面积AUC": safe_auc(time, baseline_corrected),
        "刺激期AUC": stage["stim_5_10s_auc"],
        "撤光后AUC": stage["post_10_20s_auc"],
        "最大值和最小值差值": round(float(np.nanmax(baseline_corrected) - np.nanmin(baseline_corrected)), 6),
        "波形标准差": round(float(np.nanstd(baseline_corrected)), 6),
        "波形变异系数": round(float(np.nanstd(baseline_corrected) / (abs(np.nanmean(baseline_corrected)) + 1e-9)), 6),
        "平滑后整体斜率": safe_slope(time, baseline_corrected),
        "恢复斜率": stage["post_10_20s_slope"],
        "是否持续升高": int(stage["stim_5_10s_mean"] > 0 and stage["post_10_20s_mean"] > 0),
        "是否瞬时升高": int(stage["stim_5_10s_max"] > PEAK_HEIGHT and stage["stim_5_10s_mean"] > 0 and stage["post_10_20s_mean"] <= 0),
        "是否持续降低": int(stage["stim_5_10s_mean"] < 0 and stage["post_10_20s_mean"] < 0),
    }
    summary.update(stage)
    summary["是否ON型"] = int(summary["ON反应幅度"] > PEAK_HEIGHT)
    summary["是否OFF型"] = int(summary["OFF反应幅度"] > PEAK_HEIGHT)
    summary["是否ON-OFF型"] = int(summary["是否ON型"] and summary["是否OFF型"])
    summary["是否抑制型"] = int(summary["刺激期相对刺激前变化量"] < -PEAK_HEIGHT)
    summary["反应类型初判"] = classify_response(summary)

    return preprocess, peak_df, valley_df, pd.DataFrame([stage]), summary


def plot_one(preprocess, peak_df, valley_df, summary):
    file_id = summary["file_id"]
    time = preprocess["时间(秒)"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time, preprocess["原始F/F0"], color="gray", alpha=0.45, label="Raw F/F0")
    axes[0].plot(time, preprocess["平滑去噪后F/F0"], color="#1f77b4", lw=1.7, label="Smoothed F/F0")
    if len(peak_df):
        axes[0].scatter(peak_df["峰值时间(秒)"], peak_df["峰值幅度(ΔF/F0)"] + 1, color="#d62728", s=35, label="Peaks", zorder=3)
    if len(valley_df):
        axes[0].scatter(valley_df["波谷时间(秒)"], valley_df["波谷幅度(ΔF/F0)"] + 1, color="#2ca02c", s=35, label="Valleys", zorder=3)
    axes[0].axvspan(STIM_ON, STIM_OFF, color="#f2c14e", alpha=0.22, label="Stim 5-10s")
    axes[0].axhline(1, color="black", lw=0.8, alpha=0.55)
    axes[0].set_ylabel("F/F0")
    axes[0].set_title(f"{file_id} preprocessing and events")
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, preprocess["原始dF/F0"], color="gray", alpha=0.45, label="Raw dF/F0")
    axes[1].plot(time, preprocess["基线校正后ΔF/F0"], color="#ff7f0e", lw=1.7, label="Baseline corrected")
    if len(peak_df):
        axes[1].scatter(peak_df["峰值时间(秒)"], peak_df["峰值幅度(ΔF/F0)"], color="#d62728", s=35, zorder=3)
    if len(valley_df):
        axes[1].scatter(valley_df["波谷时间(秒)"], valley_df["波谷幅度(ΔF/F0)"], color="#2ca02c", s=35, zorder=3)
    axes[1].axvspan(STIM_ON, STIM_OFF, color="#f2c14e", alpha=0.22)
    axes[1].axhline(0, color="black", lw=0.8, alpha=0.55)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("ΔF/F0")
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{file_id}_预处理峰谷图.png", dpi=180)
    plt.close()


def main():
    SINGLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(LONG_CSV)
    preprocess_all = []
    peak_all = []
    valley_all = []
    stage_all = []
    summary_all = []

    total = raw["file_id"].nunique()
    for i, (file_id, group) in enumerate(raw.groupby("file_id", sort=True), start=1):
        preprocess, peak_df, valley_df, stage_df, summary = preprocess_one(group)
        preprocess_all.append(preprocess)
        peak_all.append(peak_df)
        valley_all.append(valley_df)
        stage_all.append(stage_df)
        summary_all.append(summary)
        plot_one(preprocess, peak_df, valley_df, summary)
        if i % 25 == 0 or i == total:
            print(f"已处理 {i}/{total} 个文件")

    preprocess_df = pd.concat(preprocess_all, ignore_index=True)
    peak_df = pd.concat([x for x in peak_all if len(x)], ignore_index=True) if any(len(x) for x in peak_all) else pd.DataFrame()
    valley_df = pd.concat([x for x in valley_all if len(x)], ignore_index=True) if any(len(x) for x in valley_all) else pd.DataFrame()
    stage_df = pd.concat(stage_all, ignore_index=True)
    summary_df = pd.DataFrame(summary_all)

    preprocess_df.to_csv(PREPROCESS_CSV, index=False, encoding="utf-8-sig")
    peak_df.to_csv(PEAK_CSV, index=False, encoding="utf-8-sig")
    valley_df.to_csv(VALLEY_CSV, index=False, encoding="utf-8-sig")
    stage_df.to_csv(STAGE_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    core_cols = [
        "file_id", "文件名", "刺激前是否有波峰", "刺激前是否有波谷",
        "stim_5_10s_peak_count", "post_10_20s_peak_count", "post_10_20s_valley_count",
        "最大波峰幅度", "最大波峰出现时间", "ON反应幅度", "ON反应潜伏期",
        "OFF反应幅度", "OFF反应潜伏期", "ON/OFF比值",
        "刺激期相对刺激前变化量", "撤光后相对刺激前变化量",
        "刺激期AUC", "撤光后AUC", "波形标准差", "恢复斜率", "反应类型初判",
    ]
    summary_df[[c for c in core_cols if c in summary_df.columns]].to_csv(CORE_FEATURE_CSV, index=False, encoding="utf-8-sig")

    plot_overviews(preprocess_df, summary_df)

    print(f"输出文件夹: {OUTPUT_DIR}")
    print(f"整体特征表: {SUMMARY_CSV}")
    print(f"核心特征表: {CORE_FEATURE_CSV}")


def plot_overviews(preprocess_df, summary_df):
    pivot = preprocess_df.pivot(index="file_id", columns="时间(秒)", values="基线校正后ΔF/F0")
    time = pivot.columns.to_numpy(dtype=float)
    vals = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(11, 6))
    for row in vals:
        plt.plot(time, row, color="#6b8ba4", alpha=0.12, lw=0.8)
    mean = np.nanmean(vals, axis=0)
    plt.plot(time, mean, color="#d62728", lw=2.5, label=f"Mean (n={len(vals)})")
    plt.axvspan(STIM_ON, STIM_OFF, color="#f2c14e", alpha=0.25, label="Stim 5-10s")
    plt.axhline(0, color="black", lw=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Baseline corrected ΔF/F0")
    plt.title("All preprocessed waveforms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OVERVIEW_DIR / "全部预处理波形叠加.png", dpi=220)
    plt.close()

    heat_order = summary_df.sort_values("刺激期相对刺激前变化量", ascending=False)["file_id"].tolist()
    heat = pivot.loc[heat_order]
    plt.figure(figsize=(10, 9))
    plt.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-0.08, vmax=0.08, extent=[time.min(), time.max(), len(heat), 0])
    plt.axvspan(STIM_ON, STIM_OFF, color="#f2c14e", alpha=0.22)
    plt.colorbar(label="Baseline corrected ΔF/F0")
    plt.xlabel("Time (s)")
    plt.ylabel("Cells")
    plt.title("Preprocessed waveform heatmap")
    plt.tight_layout()
    plt.savefig(OVERVIEW_DIR / "预处理波形热图_按刺激响应排序.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    axes[0].hist(summary_df["总波峰数量"], bins=range(int(summary_df["总波峰数量"].max()) + 2), color="#4e79a7")
    axes[0].set_title("Total peak count")
    axes[1].hist(summary_df["总波谷数量"], bins=range(int(summary_df["总波谷数量"].max()) + 2), color="#59a14f")
    axes[1].set_title("Total valley count")
    axes[2].hist(summary_df["ON反应幅度"], bins=25, color="#e15759")
    axes[2].set_title("ON response amplitude")
    axes[3].hist(summary_df["撤光后相对刺激前变化量"], bins=25, color="#f28e2b")
    axes[3].set_title("Post-stimulus delta")
    for ax in axes:
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OVERVIEW_DIR / "核心特征分布.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
