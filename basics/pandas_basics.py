# -*- coding: utf-8 -*-
"""
Pandas 基础：Series、DataFrame、读取 csv、常用统计
运行：python basics/pandas_basics.py
"""
import os

import pandas as pd


def demo_series():
    """1. Series：带索引的一维数据"""
    s = pd.Series([88, 92, 75, 60], index=["语文", "数学", "英语", "体育"], name="成绩")
    print("Series:\n", s)
    print("最大值科目:", s.idxmax(), "| 平均分:", round(s.mean(), 1))
    print("及格的科目:", s[s >= 60].index.tolist())


def demo_dataframe():
    """2. DataFrame：二维表格，核心数据结构"""
    df = pd.DataFrame({
        "姓名": ["张三", "李四", "王五", "赵六"],
        "部门": ["技术", "运营", "技术", "市场"],
        "工资": [15000, 9000, 18000, 8000],
    })
    print("\nDataFrame:\n", df)
    print("shape:", df.shape, "| 列名:", df.columns.tolist())
    print("\n按部门平均工资:\n", df.groupby("部门")["工资"].mean().round(0).to_string())
    print("\n工资 > 1 万的人:", df.loc[df["工资"] > 10000, "姓名"].tolist())


def demo_read_csv():
    """3. 读取项目自带的示例 csv（先运行 data/generate_data.py 生成）"""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "users.csv")
    if not os.path.exists(csv_path):
        print("\n[跳过] 未找到 users.csv，先运行：python data/generate_data.py")
        return
    df = pd.read_csv(csv_path)
    print("\n用户数据前 5 行:\n", df.head().to_string())
    print("\ninfo 概览：", df.shape[0], "行 ×", df.shape[1], "列")
    print("各列缺失值:\n", df.isna().sum().to_string())
    print("\n数值列统计:\n", df.describe().round(1).to_string())


if __name__ == "__main__":
    demo_series()
    demo_dataframe()
    demo_read_csv()
