# -*- coding: utf-8 -*-
"""
Matplotlib 可视化：折线图、柱状图、直方图
中文显示：Windows 指定微软雅黑；保存到 output/ 目录
运行：python analysis/visualize.py（需先生成并清洗数据）
"""
import os

import matplotlib

matplotlib.use("Agg")  # 无界面环境直接保存文件
import matplotlib.pyplot as plt
import pandas as pd

from data_cleaning import load_and_clean

# 中文字体设置（Windows 自带微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 修复负号显示

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def plot_churn_by_tenure(df):
    """折线图：不同在网时长的流失率变化趋势"""
    rate = df.groupby("tenure_months")["churn"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rate.index, rate.values, marker="o", markersize=4, color="#185FA5")
    ax.set_xlabel("在网时长（月）")
    ax.set_ylabel("流失率（%）")
    ax.set_title("用户流失率随在网时长的变化")
    ax.grid(alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "line_流失率趋势.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_city_spend(df):
    """柱状图：各城市平均月消费对比"""
    avg = df.groupby("city")["monthly_spend"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(avg.index, avg.values, color="#378ADD")
    for b, v in zip(bars, avg.values):  # 柱顶标数值
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("平均月消费（元）")
    ax.set_title("各城市用户平均月消费")
    path = os.path.join(OUTPUT_DIR, "bar_城市消费.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_age_hist(df):
    """直方图：年龄分布"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["age"], bins=20, color="#97C459", edgecolor="white")
    ax.set_xlabel("年龄")
    ax.set_ylabel("人数")
    ax.set_title("用户年龄分布")
    path = os.path.join(OUTPUT_DIR, "hist_年龄分布.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_all_charts(df=None):
    """一次生成全部图表，返回文件路径列表"""
    if df is None:
        df = load_and_clean()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return [
        plot_churn_by_tenure(df),
        plot_city_spend(df),
        plot_age_hist(df),
    ]


if __name__ == "__main__":
    for p in generate_all_charts():
        print("图表已生成:", os.path.abspath(p))
