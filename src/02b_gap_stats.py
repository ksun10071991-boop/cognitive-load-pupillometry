"""
02b_gap_stats.py
----------------
缺失情况诊断:在决定插补规则之前,先量化缺口的长度和每个 block 的缺失比例。
不修改数据,只打印统计结果。
"""

import pandas as pd

CONFIDENCE_THRESHOLD = 0.8

df = pd.read_parquet("../data/colet.parquet")
df = df.sort_values(["Participant", "Task", "Timestamp"]).reset_index(drop=True)
df["is_missing"] = df["Confidence"] < CONFIDENCE_THRESHOLD

# --- 1. 采样间隔:相邻两点隔多久?用来把"缺了几个点"换算成"缺了多少秒" ---
dt = df.groupby(["Participant", "Task"])["Timestamp"].diff()
print("=== 采样间隔 ===")
print(dt.describe())

# --- 2. 每个缺口有多长(连续缺失了几个点) ---
# 思路:每当"换了 block"或者"缺/不缺的状态变了",就开始一段新的 run,给它一个新编号
block_change = (df["Participant"] != df["Participant"].shift()) | (df["Task"] != df["Task"].shift())
state_change = df["is_missing"] != df["is_missing"].shift()
df["run_id"] = (block_change | state_change).cumsum()

gap_lengths = df[df["is_missing"]].groupby("run_id").size()
print("\n=== 缺口长度(单位:采样点数) ===")
print("缺口总数:", len(gap_lengths))
print(gap_lengths.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

# --- 3. 每个 block 的缺失比例 ---
block_missing = df.groupby(["Participant", "Task"])["is_missing"].mean()
print("\n=== 每个 block 的缺失比例 ===")
print(block_missing.describe())
print("\n缺失最严重的 10 个 block:")
print(block_missing.sort_values(ascending=False).head(10))