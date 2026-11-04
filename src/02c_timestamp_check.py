"""
02c_timestamp_check.py
----------------------
诊断时间戳:为什么一半以上的相邻采样间隔为 0?最大的时间跳跃发生在哪?
不修改数据,只打印。
"""

import pandas as pd

df = pd.read_parquet("../data/colet.parquet")

# --- 1. 看一个 block 的原始排列(不排序,保持文件原本的顺序) ---
block = df[(df["Participant"] == "P_01") & (df["Task"] == "Task_1")]
print("=== P_01 / Task_1 原始前 20 行 ===")
print(block.head(20))

print("\n该 block 行数:", len(block))
print("不重复的时间戳个数:", block["Timestamp"].nunique())

# 每个时间戳出现了几次?(如果大部分是 2 次,双眼假说就很可能成立)
print("\n=== 同一个时间戳出现的次数分布 ===")
print(block["Timestamp"].value_counts().value_counts())

# --- 2. 全数据里最大的 5 次时间跳跃,发生在哪个 block ---
df_sorted = df.sort_values(["Participant", "Task", "Timestamp"])
dt = df_sorted.groupby(["Participant", "Task"])["Timestamp"].diff()
biggest = dt.sort_values(ascending=False).head(5)

print("\n=== 最大的 5 次时间跳跃 ===")
print(df_sorted.loc[biggest.index, ["Participant", "Task", "Timestamp"]].assign(jump=biggest))