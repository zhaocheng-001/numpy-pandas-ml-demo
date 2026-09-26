# -*- coding: utf-8 -*-
"""
数据清洗：缺失值处理、重复值处理、异常值检查
运行：python analysis/data_cleaning.py
"""
import os

import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "users.csv")


def clean(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    三步清洗，返回清洗后的副本（不修改原数据）：
    1. 去重复：user_id 重复的行只保留第一条
    2. 填缺失：数值列用中位数填充（抗极端值），城市列缺失直接删行
    3. 查异常：年龄/消费金额超出合理范围打日志
    """
    report = {}
    df = df.copy()  # 显式复制，避免 SettingWithCopyWarning

    # ---- 1. 去重复 ----
    before = len(df)
    df = df.drop_duplicates(subset="user_id", keep="first")
    report["删除重复行"] = before - len(df)

    # ---- 2. 缺失值 ----
    report["各列缺失"] = df.isna().sum().to_dict()
    for col in ["age", "monthly_spend", "visit_count"]:
        median = df[col].median()
        df[col] = df[col].fillna(median)
        if verbose:
            print(f"  [填充] {col} 缺失 {report['各列缺失'][col]} 个 → 中位数 {median}")

    # 分类列（city）缺失：整行删除（样本充足时常用策略）
    city_missing = df["city"].isna().sum()
    if city_missing:
        df = df.dropna(subset=["city"])
        if verbose:
            print(f"  [删行] city 缺失 {city_missing} 行已删除")

    # ---- 3. 异常值检查（打印出来，不改动）----
    bad_age = ((df["age"] < 16) | (df["age"] > 70)).sum()
    bad_spend = (df["monthly_spend"] <= 0).sum()
    report["异常值"] = {"年龄越界": int(bad_age), "消费非正": int(bad_spend)}
    if bad_age or bad_spend:
        print(f"  [警告] 发现异常值：{report['异常值']}")

    if verbose:
        for k, v in report.items():
            if k != "各列缺失":
                print(f"  {k}: {v}")
    return df.reset_index(drop=True)


def load_and_clean(path=DATA_PATH):
    """读取 csv → 清洗，完整入口"""
    df = pd.read_csv(path)
    print(f"原始数据：{df.shape[0]} 行")
    df = clean(df)
    print(f"清洗后：{df.shape[0]} 行，剩余缺失值 {int(df.isna().sum().sum())} 个")
    return df


if __name__ == "__main__":
    load_and_clean()
