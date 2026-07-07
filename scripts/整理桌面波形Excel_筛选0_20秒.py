from pathlib import Path
import hashlib
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_ROOT / "data" / "raw_excel_all"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "screened_0_20_excel"
INCLUDED_DIR = OUTPUT_ROOT / "included_excel"
EXCLUDED_DIR = OUTPUT_ROOT / "剔除文件"
EXCLUDED_DUP_DIR = EXCLUDED_DIR / "重复文件"
EXCLUDED_TIME_DIR = EXCLUDED_DIR / "非0-20秒文件"
EXCLUDED_FORMAT_DIR = EXCLUDED_DIR / "非原始波形Excel"
REPORT_XLSX = OUTPUT_ROOT / "0-20秒Excel整理清单.xlsx"

TIME_COLUMN = "Time - F/F0"
SIGNAL_COLUMN = "F/F0 - F/F0"

START_MAX = 0.25
END_MIN = 19.5
END_MAX = 20.5
MIN_POINTS = 100


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(path: Path, used_names: set[str]) -> str:
    name = path.name
    if name not in used_names:
        used_names.add(name)
        return name
    stem = path.stem
    suffix = path.suffix
    parent_tag = path.parent.name.replace("/", "_")
    candidate = f"{stem}__{parent_tag}{suffix}"
    i = 2
    while candidate in used_names:
        candidate = f"{stem}__{parent_tag}_{i}{suffix}"
        i += 1
    used_names.add(candidate)
    return candidate


def find_excel_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() not in {".xlsx", ".xls"}:
            continue
        if OUTPUT_ROOT in path.parents:
            continue
        files.append(path)
    return sorted(files)


def read_time_info(path: Path) -> dict:
    try:
        head = pd.read_excel(path, sheet_name=0, nrows=5)
        columns = list(head.columns)
        if TIME_COLUMN not in columns or SIGNAL_COLUMN not in columns:
            return {
                "is_raw_waveform": False,
                "read_ok": True,
                "reason": "非原始波形Excel",
            }

        full_time = pd.read_excel(path, sheet_name=0, usecols=[TIME_COLUMN])[TIME_COLUMN]
        time = pd.to_numeric(full_time, errors="coerce").dropna().to_numpy()
        if len(time) < 2:
            return {
                "is_raw_waveform": True,
                "read_ok": True,
                "reason": "时间点过少",
                "n_points": int(len(time)),
                "is_0_20": False,
            }

        start = float(time.min())
        end = float(time.max())
        median_dt = float(pd.Series(time).diff().dropna().median())
        is_0_20 = (
            start <= START_MAX
            and END_MIN <= end <= END_MAX
            and len(time) >= MIN_POINTS
        )
        return {
            "is_raw_waveform": True,
            "read_ok": True,
            "reason": "纳入" if is_0_20 else "非0-20秒文件",
            "n_points": int(len(time)),
            "start_time": start,
            "end_time": end,
            "median_dt": median_dt,
            "is_0_20": bool(is_0_20),
        }
    except Exception as exc:
        return {
            "is_raw_waveform": False,
            "read_ok": False,
            "reason": f"读取失败: {exc}",
        }


def copy_file(src: Path, dst_dir: Path, used_names: set[str]) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_name = safe_name(src, used_names)
    shutil.copy2(src, dst_dir / out_name)
    return out_name


def main():
    INCLUDED_DIR.mkdir(parents=True, exist_ok=True)
    EXCLUDED_DUP_DIR.mkdir(parents=True, exist_ok=True)
    EXCLUDED_TIME_DIR.mkdir(parents=True, exist_ok=True)
    EXCLUDED_FORMAT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_hashes = {}
    seen_names = {}
    included_names = set()
    excluded_dup_names = set()
    excluded_time_names = set()
    excluded_format_names = set()

    files = find_excel_files(ROOT_DIR)
    for i, path in enumerate(files, start=1):
        rel = path.relative_to(ROOT_DIR)
        info = read_time_info(path)
        digest = file_hash(path)

        duplicate_type = ""
        duplicate_of = ""
        if info.get("is_raw_waveform"):
            if digest in seen_hashes:
                duplicate_type = "内容重复"
                duplicate_of = seen_hashes[digest]
            elif path.name in seen_names:
                duplicate_type = "文件名重复"
                duplicate_of = seen_names[path.name]

        action = ""
        copied_name = ""

        if duplicate_type:
            action = "剔除_重复"
            copied_name = copy_file(path, EXCLUDED_DUP_DIR, excluded_dup_names)
        elif info.get("is_raw_waveform") and info.get("is_0_20"):
            action = "纳入"
            copied_name = copy_file(path, INCLUDED_DIR, included_names)
            seen_hashes[digest] = str(rel)
            seen_names[path.name] = str(rel)
        elif info.get("is_raw_waveform"):
            action = "剔除_非0-20秒"
            copied_name = copy_file(path, EXCLUDED_TIME_DIR, excluded_time_names)
            seen_hashes[digest] = str(rel)
            seen_names[path.name] = str(rel)
        else:
            action = "剔除_非原始波形Excel"
            copied_name = copy_file(path, EXCLUDED_FORMAT_DIR, excluded_format_names)

        rows.append(
            {
                "原路径": str(path),
                "相对路径": str(rel),
                "文件名": path.name,
                "是否原始波形Excel": info.get("is_raw_waveform", False),
                "读取成功": info.get("read_ok", False),
                "起始时间": info.get("start_time", ""),
                "结束时间": info.get("end_time", ""),
                "数据点数": info.get("n_points", ""),
                "中位时间间隔": info.get("median_dt", ""),
                "是否0-20秒": info.get("is_0_20", False),
                "重复类型": duplicate_type,
                "重复来源": duplicate_of,
                "处理动作": action,
                "复制后文件名": copied_name,
                "原因": info.get("reason", ""),
            }
        )

        if i % 50 == 0 or i == len(files):
            print(f"已处理 {i}/{len(files)} 个 Excel")

    report = pd.DataFrame(rows)
    summary = (
        report.groupby("处理动作")
        .size()
        .reset_index(name="数量")
        .sort_values("处理动作")
    )

    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="汇总", index=False)
        report.to_excel(writer, sheet_name="整理明细", index=False)
        report[report["处理动作"] == "纳入"].to_excel(writer, sheet_name="纳入文件", index=False)
        report[report["处理动作"] != "纳入"].to_excel(writer, sheet_name="剔除文件", index=False)

    readme = OUTPUT_ROOT / "整理说明.md"
    readme.write_text(
        "\n".join(
            [
                "# 0-20 秒原始 Excel 整理说明",
                "",
                "整理目录：`/Users/tangdi/Desktop/波形Excel数据`",
                "",
                "纳入标准：",
                "",
                "- 原始波形 Excel，表头包含 `Time - F/F0` 和 `F/F0 - F/F0`",
                "- 起始时间在 0 秒附近",
                "- 结束时间在 19.5-20.5 秒之间",
                "- 数据点数不少于 100",
                "- 不和已经纳入的原始波形 Excel 重复",
                "",
                "文件夹说明：",
                "",
                "- `纳入_0-20秒Excel`：最终纳入的新总 Excel 文件夹",
                "- `剔除文件/非0-20秒文件`：原始波形 Excel，但时间范围不符合 0-20 秒",
                "- `剔除文件/非原始波形Excel`：处理结果、汇总表等，不是原始波形 Excel",
                "- `剔除文件/重复文件`：重复原始波形 Excel",
                "- `0-20秒Excel整理清单.xlsx`：完整整理明细和汇总",
                "",
                "本次整理只复制文件，不物理删除原始文件。",
                "",
                summary.to_string(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"输出文件夹: {OUTPUT_ROOT}")
    print(f"纳入文件夹: {INCLUDED_DIR}")
    print(f"剔除文件夹: {EXCLUDED_DIR}")
    print(f"整理清单: {REPORT_XLSX}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
