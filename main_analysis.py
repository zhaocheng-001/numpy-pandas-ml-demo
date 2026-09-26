# -*- coding: utf-8 -*-
"""
完整小案例主线：数据生成 → 读取 → 清洗 → 可视化 → 建模 → 评估 → 导出结果
一条命令跑通全流程：python main_analysis.py
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from generate_data import generate_users  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")


def main():
    os.makedirs(OUT, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 60)
    print("步骤 1/5 生成示例数据（含缺失值、重复行）")
    print("=" * 60)
    csv_path = os.path.join(BASE, "data", "users.csv")
    if not os.path.exists(csv_path):
        generate_users().to_csv(csv_path, index=False, encoding="utf-8-sig")
    raw = pd.read_csv(csv_path)
    print(f"原始数据：{raw.shape[0]} 行，缺失值 {int(raw.isna().sum().sum())} 个，"
          f"重复行 {int(raw.duplicated(subset='user_id').sum())} 条")

    print("=" * 60)
    print("步骤 2/5 数据清洗")
    print("=" * 60)
    from data_cleaning import clean
    df = clean(raw)
    df.to_csv(os.path.join(OUT, "users_cleaned.csv"), index=False, encoding="utf-8-sig")
    print(f"清洗后保存到 output/users_cleaned.csv（{df.shape[0]} 行）")

    print("=" * 60)
    print("步骤 3/5 可视化")
    print("=" * 60)
    from visualize import generate_all_charts
    charts = generate_all_charts(df)
    for p in charts:
        print("  图表：", os.path.relpath(p, BASE))

    print("=" * 60)
    print("步骤 4/5 逻辑回归建模")
    print("=" * 60)
    from ml_demo import train_churn_model
    model, metrics = train_churn_model(df)
    print(f"测试集准确率：{metrics['accuracy']:.1%}")
    print(metrics["report"])

    print("=" * 60)
    print("步骤 5/5 导出评估结果")
    print("=" * 60)
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metrics.items() if k != "report"},
                  f, ensure_ascii=False, indent=2)
    print("指标已保存到 output/metrics.json")
    print("全部完成！图表见 output/，分析结论见 docs/分析报告.md")


if __name__ == "__main__":
    main()
