# -*- coding: utf-8 -*-
"""
sklearn 入门：逻辑回归预测用户是否流失
技术点：数据集划分（train_test_split）、特征标准化（Pipeline）、模型训练与评估
运行：python analysis/ml_demo.py
"""
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_cleaning import load_and_clean

FEATURES = ["age", "monthly_spend", "visit_count", "tenure_months"]


def train_churn_model(df=None, seed=42):
    """
    完整建模流程：
    1. 选特征 X 和标签 y
    2. 划分训练集(80%) / 测试集(20%)，stratify 保证流失比例一致
    3. Pipeline = 标准化 + 逻辑回归
    4. 训练并在测试集上评估
    返回 (模型, 评估指标 dict)
    """
    if df is None:
        df = load_and_clean()

    X, y = df[FEATURES], df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),                    # 特征标准化：不同量纲统一到 0-1 量级
        ("clf", LogisticRegression(max_iter=1000)),      # 逻辑回归（二分类）
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "report": classification_report(y_test, y_pred,
                                        target_names=["留存", "流失"], digits=3),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    # 打印各特征的影响系数（标准化后可直接比较大小）
    coefs = model.named_steps["clf"].coef_[0]
    print("特征影响系数（正=促进流失，负=促进留存）:")
    for name, c in sorted(zip(FEATURES, coefs), key=lambda t: -abs(t[1])):
        print(f"  {name:15s} {c:+.3f}")
    return model, metrics


if __name__ == "__main__":
    model, m = train_churn_model()
    print(f"\n训练集 {m['train_rows']} 行 / 测试集 {m['test_rows']} 行")
    print(f"测试集准确率 accuracy = {m['accuracy']:.1%}\n")
    print(m["report"])
