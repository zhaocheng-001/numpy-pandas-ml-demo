# -*- coding: utf-8 -*-
"""
冒烟测试：Numpy/Pandas/清洗/可视化/建模 全链路（不依赖外部文件，数据现场生成）
运行：python tests/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/analysis")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

passed = failed = 0


def check(name, fn):
    global passed, failed
    try:
        result = fn()
        passed += 1
        print(f"[PASS] {name} -> {result}")
    except Exception as e:
        failed += 1
        print(f"[FAIL] {name} -> {type(e).__name__}: {e}")


def test_numpy():
    import numpy as np
    a = np.arange(10).reshape(2, 5)
    assert (a * 2 + 1 == np.arange(1, 21, 2).reshape(2, 5)).all()  # 向量化
    assert a[:, 1].tolist() == [1, 6]                              # 切片
    assert a[a > 4].tolist() == [5, 6, 7, 8, 9]                    # 布尔索引
    return "向量化/切片/布尔索引 OK"


def test_generate_data():
    from data.generate_data import generate_users
    df = generate_users(500)
    assert len(df) > 500                    # 注入了重复行
    assert df.isna().sum().sum() > 0        # 注入了缺失值
    assert set(df["churn"].unique()) <= {0, 1}
    return f"{len(df)} 行，流失率 {df['churn'].mean():.1%}"


def test_clean():
    import pandas as pd
    from analysis.data_cleaning import clean
    df = pd.DataFrame({
        "user_id": [1, 2, 2, 3],
        "age": [30, None, 40, 50],
        "city": ["北京", "上海", "广州", None],
        "monthly_spend": [100, None, 300, 400],
        "visit_count": [5, 6, None, 8],
        "tenure_months": [1, 2, 3, 4],
        "churn": [0, 1, 0, 1],
    })
    out = clean(df, verbose=False)
    assert len(out) == 2                          # 去重 1 条 + city 缺失删 1 行
    assert out.isna().sum().sum() == 0            # 无残留缺失
    return f"4 行脏数据 → 清洗后 {len(out)} 行、0 缺失"


def test_model():
    from analysis.data_cleaning import load_and_clean
    from analysis.ml_demo import train_churn_model
    csv = os.path.join(os.path.dirname(__file__), "..", "data", "users.csv")
    if not os.path.exists(csv):
        from data.generate_data import generate_users
        generate_users().to_csv(csv, index=False, encoding="utf-8-sig")
    df = load_and_clean(csv)
    _, metrics = train_churn_model(df)
    acc = metrics["accuracy"]
    assert acc >= 0.65, f"准确率过低：{acc}"
    return f"逻辑回归准确率 {acc:.1%}"


def test_charts():
    from analysis.data_cleaning import load_and_clean
    from analysis.visualize import generate_all_charts
    csv = os.path.join(os.path.dirname(__file__), "..", "data", "users.csv")
    df = load_and_clean(csv)
    paths = generate_all_charts(df)
    assert len(paths) == 3 and all(os.path.exists(p) for p in paths)
    return f"3 张图表生成 OK"


if __name__ == "__main__":
    check("Numpy 向量化/切片", test_numpy)
    check("示例数据生成", test_generate_data)
    check("数据清洗", test_clean)
    check("逻辑回归建模", test_model)
    check("Matplotlib 图表", test_charts)
    print(f"\n结果：{passed} 通过，{failed} 失败")
    sys.exit(1 if failed else 0)
