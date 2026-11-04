"""
03_label.py
-----------
打标签:把每个 block 的 NASA_RTLX 分数转换成三类认知负荷标签。

规则(项目要求): NASA < 30 → 0 低 | 30 <= NASA < 50 → 1 中 | NASA >= 50 → 2 高

输入: data/colet_preprocessed.parquet
输出: data/colet_labeled.parquet (多了一列 Label)
"""

import pandas as pd

LOW_MAX = 30    # 低于 30 分是"低"
MED_MAX = 50    # 30 到 50 之间是"中",50 及以上是"高"
LABEL_NAMES = {0: "low", 1: "medium", 2: "high"}

df = pd.read_parquet("../data/colet_preprocessed.parquet")

# 每个 block 只有一个 NASA 分数,先抽出一张"block 名单"(186 行)
blocks = df.groupby(["Participant", "Task"])["NASA_RTLX"].first().reset_index()

# --- 体检:分数有没有缺?是不是整数? ---
print("缺失 NASA 分数的 block:", blocks["NASA_RTLX"].isna().sum())
print("非整数分数的 block 数:", (blocks["NASA_RTLX"] % 1 != 0).sum())
print("分数范围:", blocks["NASA_RTLX"].min(), "到", blocks["NASA_RTLX"].max())


# --- 贴标签 ---
def score_to_label(score):
    if score < LOW_MAX:
        return 0
    elif score < MED_MAX:
        return 1
    else:
        return 2


blocks["Label"] = blocks["NASA_RTLX"].apply(score_to_label)

print("\n=== 每个类别有几个 block ===")
print(blocks["Label"].map(LABEL_NAMES).value_counts())

print("\n=== 每个 Task 的标签分布 ===")
print(pd.crosstab(blocks["Task"], blocks["Label"].map(LABEL_NAMES)))

# --- 把标签贴回每一个格子上,保存 ---
df = df.merge(blocks[["Participant", "Task", "Label"]],
              on=["Participant", "Task"], how="left")
df.to_parquet("../data/colet_labeled.parquet", index=False)
print("\n已保存: data/colet_labeled.parquet")