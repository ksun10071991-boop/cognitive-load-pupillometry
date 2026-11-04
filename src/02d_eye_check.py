"""
02d_eye_check.py
----------------
验证假说:同一时间戳的两行是同一测量的重复;相邻时间戳是左右眼交替。
不修改数据,只打印。
"""

import pandas as pd

CONFIDENCE_THRESHOLD = 0.8

df = pd.read_parquet("../data/colet.parquet")
df = df[df["Confidence"] >= CONFIDENCE_THRESHOLD]   # 只看好点,避免眨眼干扰判断

g = df.groupby(["Participant", "Task", "Timestamp"])["Pupil_Diameter"]

# 1. 同一时间戳内部的差异(假说:很小)
within = g.max() - g.min()
print("=== 同一时间戳内,两行的差值 ===")
print(within.describe())

# 每个时间戳取一个平均值,按时间排好
per_ts = g.mean().reset_index().sort_values(["Participant", "Task", "Timestamp"])
grp = per_ts.groupby(["Participant", "Task"])["Pupil_Diameter"]

# 2. 相邻时间戳之间的差异(假说:较大,因为换了一只眼)
per_ts["diff_lag1"] = grp.diff(1).abs()
# 3. 隔一个时间戳的差异(假说:很小,因为又回到同一只眼)
per_ts["diff_lag2"] = grp.diff(2).abs()

print("\n=== 相邻时间戳之差(lag 1) ===")
print(per_ts["diff_lag1"].describe())
print("\n=== 隔一个时间戳之差(lag 2) ===")
print(per_ts["diff_lag2"].describe())