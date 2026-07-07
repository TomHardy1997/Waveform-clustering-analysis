from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "included_excel"
REPORT_XLSX = PROJECT_ROOT / "data" / "metadata" / "0-20秒Excel整理清单.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "results" / "visualization"
LONG_CSV = OUTPUT_DIR / "纳入文件_0-20秒原始波形长表.csv"
META_CSV = OUTPUT_DIR / "纳入文件来源信息.csv"

TIME_COLUMN = "Time - F/F0"
SIGNAL_COLUMN = "F/F0 - F/F0"
DFF_COLUMN = "F/F0 - dF/F0"


def load_source_map() -> dict:
    if not REPORT_XLSX.exists():
        return {}
    report = pd.read_excel(REPORT_XLSX, sheet_name="纳入文件")
    source_map = {}
    for _, row in report.iterrows():
        copied = str(row.get("复制后文件名", "")).strip()
        if copied:
            source_map[copied] = {
                "source_path": row.get("原路径", ""),
                "source_relative_path": row.get("相对路径", ""),
                "source_start_time": row.get("起始时间", ""),
                "source_end_time": row.get("结束时间", ""),
                "source_n_points": row.get("数据点数", ""),
            }
    return source_map


def read_one_file(path: Path, source_map: dict) -> tuple[pd.DataFrame, dict]:
    df = pd.read_excel(path, sheet_name=0)
    if TIME_COLUMN not in df.columns or SIGNAL_COLUMN not in df.columns:
        raise ValueError(f"缺少原始波形列: {path.name}")

    out = pd.DataFrame(
        {
            "time": pd.to_numeric(df[TIME_COLUMN], errors="coerce"),
            "f_f0": pd.to_numeric(df[SIGNAL_COLUMN], errors="coerce"),
        }
    )
    if DFF_COLUMN in df.columns:
        out["dff"] = pd.to_numeric(df[DFF_COLUMN], errors="coerce")
    else:
        out["dff"] = pd.NA

    out = out.dropna(subset=["time", "f_f0"])
    out = out[(out["time"] >= 0) & (out["time"] <= 20)].copy()
    out = out.sort_values("time").drop_duplicates("time", keep="first")

    file_id = path.stem
    source = source_map.get(path.name, {})

    out.insert(0, "file_id", file_id)
    out.insert(1, "filename", path.name)
    out.insert(2, "source_relative_path", source.get("source_relative_path", ""))

    time = out["time"]
    meta = {
        "file_id": file_id,
        "filename": path.name,
        "current_path": str(path),
        "source_path": source.get("source_path", ""),
        "source_relative_path": source.get("source_relative_path", ""),
        "n_points": int(len(out)),
        "start_time": float(time.min()) if len(time) else None,
        "end_time": float(time.max()) if len(time) else None,
        "median_dt": float(time.diff().dropna().median()) if len(time) > 1 else None,
    }
    return out, meta


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_map = load_source_map()
    files = sorted(INPUT_DIR.glob("*.xlsx")) + sorted(INPUT_DIR.glob("*.xls"))

    all_rows = []
    meta_rows = []
    errors = []
    for i, path in enumerate(files, start=1):
        try:
            raw, meta = read_one_file(path, source_map)
            all_rows.append(raw)
            meta_rows.append(meta)
        except Exception as exc:
            errors.append({"filename": path.name, "path": str(path), "error": str(exc)})

        if i % 25 == 0 or i == len(files):
            print(f"已读取 {i}/{len(files)} 个纳入 Excel")

    pd.concat(all_rows, ignore_index=True).to_csv(LONG_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(meta_rows).to_csv(META_CSV, index=False, encoding="utf-8-sig")

    if errors:
        pd.DataFrame(errors).to_csv(OUTPUT_DIR / "读取失败文件.csv", index=False, encoding="utf-8-sig")

    print(f"原始波形长表: {LONG_CSV}")
    print(f"来源信息: {META_CSV}")
    print(f"成功读取: {len(meta_rows)} 个")
    if errors:
        print(f"读取失败: {len(errors)} 个")


if __name__ == "__main__":
    main()
