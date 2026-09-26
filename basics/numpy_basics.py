# -*- coding: utf-8 -*-
"""
NumPy 基础：数组创建、向量化运算、切片与布尔索引
运行：python basics/numpy_basics.py
"""
import numpy as np


def demo_create():
    """1. 数组创建的常用方式"""
    a = np.array([1, 2, 3, 4, 5])              # 从列表创建
    b = np.zeros((2, 3))                       # 全 0 二维数组
    c = np.arange(0, 10, 2)                    # 等差序列 [0,2,4,6,8]
    d = np.linspace(0, 1, 5)                   # 等分序列 5 个点
    rng = np.random.default_rng(42)            # 随机数生成器（固定种子可复现）
    e = rng.integers(1, 7, size=(3, 3))        # 3x3 模拟骰子
    print("一维数组:", a, "| dtype:", a.dtype)
    print("全0数组 shape:", b.shape, "| arange:", c, "| linspace:", d.round(2))
    print("随机矩阵:\n", e)
    return a


def demo_vectorize(a):
    """2. 向量化运算：不用写 for 循环，整组数字一起算（比循环快几十倍）"""
    print("原始数组      :", a)
    print("加法 a + 10   :", a + 10)            # 广播：标量作用到每个元素
    print("乘法 a * 2    :", a * 2)
    print("统计 mean/max :", a.mean(), a.max())
    print("平方和开根号  :", np.sqrt((a ** 2).sum()).round(3))

    # 向量化 vs Python 循环的性能对比
    big = np.arange(1_000_000)
    import time
    t0 = time.perf_counter()
    _ = big * 2 + 1                      # 向量化：一行
    t_vec = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = [x * 2 + 1 for x in big]         # 列表推导：逐个算
    t_loop = time.perf_counter() - t0
    print(f"100 万元素：向量化 {t_vec*1000:.1f}ms vs 循环 {t_loop*1000:.1f}ms，"
          f"快 {t_loop/max(t_vec,1e-9):.0f} 倍")


def demo_slice(a):
    """3. 切片与布尔索引"""
    m = np.arange(1, 13).reshape(3, 4)   # 3 行 4 列矩阵
    print("矩阵:\n", m)
    print("第 1 行       :", m[0])
    print("第 2 列       :", m[:, 1])
    print("左上 2x2 子块 :\n", m[:2, :2])
    print("布尔索引 >6   :", m[m > 6])        # 筛出所有大于 6 的元素
    m[m > 6] = 0                          # 条件赋值
    print("大于 6 置 0 后:\n", m)


if __name__ == "__main__":
    a = demo_create()
    print("-" * 50)
    demo_vectorize(a)
    print("-" * 50)
    demo_slice(a)
