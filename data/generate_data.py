# -*- coding: utf-8 -*-
"""
示例数据生成：模拟 1000 名电商用户的流失数据
- 故意注入 5% 缺失值和 3% 重复行，供数据清洗练习
- 流失概率与消费金额/在网时长相关，保证逻辑回归能学到真实规律
运行：python data/generate_data.py
"""
import os

import numpy as np
import pandas as pd

SEED = 42
N = 1000


def generate_users(n=N, seed=SEED):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "age": rng.normal(35, 9, n).clip(18, 60).round(0),
        "city": rng.choice(["北京", "上海", "广州", "深圳", "杭州"], n, p=[0.25, 0.25, 0.2, 0.15, 0.15]),
        "monthly_spend": np.round(np.exp(rng.normal(5.5, 0.6, n)), 2),   # 月消费(元)，对数正态
        "visit_count": rng.poisson(12, n),                               # 月访问次数
        "tenure_months": rng.integers(1, 37, n),                         # 在网月数 1-36
    })

    # 流失规律：消费越低、在网越短 → 越容易流失（给模型留出可学习的信号）
    # 用标准化后的特征构造线性分数，信号强度可控、方向明确
    z = lambda s: (s - s.mean()) / s.std()
    score = (1.3 * z(df["monthly_spend"])      # 消费高 → 留存
             + 1.2 * z(df["tenure_months"])    # 在网久 → 留存
             - 0.3 * z(df["age"])              # 年长 → 略易流失
             + rng.normal(0, 0.3, n))         # 少量随机噪声
    prob = 1 / (1 + np.exp(score))  # sigmoid：score 越大流失概率越低
    df["churn"] = (rng.random(n) < prob).astype(int)

    # ---- 注入脏数据：缺失值 + 重复行 ----
    for col in ["age", "monthly_spend", "visit_count"]:
        missing_idx = rng.choice(n, size=int(n * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan

    dup_rows = df.sample(int(n * 0.03), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)  # 打乱顺序


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_users()
    path = os.path.join(out_dir, "users.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已生成 {path}：{df.shape[0]} 行（含 {df.isna().sum().sum()} 个缺失值、"
          f"{df.duplicated(subset='user_id').sum()} 条重复）")
    print(df["churn"].value_counts().rename("人数").to_string())
