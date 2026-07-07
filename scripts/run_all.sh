#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python scripts/提取纳入0_20秒波形数据.py
python scripts/可视化纳入0_20秒波形.py
python scripts/批量处理纳入0_20秒_预处理特征.py
python scripts/生成纳入0_20秒_处理结果Excel和Notebook.py
python scripts/生成0_20秒波形聚类分析.py
