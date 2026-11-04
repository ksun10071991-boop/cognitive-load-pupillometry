"""
04_features.py
--------------
特征工程:按受试者归一化 → 滑动窗口切段 → 每段计算统计特征。

归一化: 每个受试者用自己全部数据的均值和标准差做 z-score。
        按人而不是按 block,是为了保留任务之间的瞳孔差异(这正是要学的信号)。
        只用受试者自己的瞳孔数据,不用标签,不用他人数据,不造成泄露。
窗口:   10 秒一段,每次移动 5 秒(50% 重叠)。暂定值,写入 README。
        含 NaN(长缺口)的窗口整段丢弃。
特征:   mean, std, min, max, slope

输入: data/colet_labeled.parquet
输出: data/features.parquet
"""

import numpy as np
import pandas as pd

BIN_SEC = 0.05
WINDOW_SEC = 10
STEP_SEC = 5
WINDOW_BINS = int(WINDOW_SEC / BIN_SEC)   # 200 个格子
STEP_BINS = int(STEP_SEC / BIN_SEC)       # 100 个格子
LABEL_NAMES = {0: "low", 1: "medium", 2: "high"}

df = pd.read_parquet("../data/colet_labeled.parquet")

# --- 先看看每个 block 有多长(秒) ---
block_len = df.groupby(["Participant", "Task"])["Time"].max()
print("=== block 时长(秒) ===")
print(block_len.describe())

# --- 第 1 步:按受试者归一化(z-score) ---
grp = df.groupby("Participant")["Pupil"]
df["Pupil_z"] = (df["Pupil"] - grp.transform("mean")) / grp.transform("std")

# --- 第 2 步 + 第 3 步:切窗口,算特征 ---
rows = []
n_dropped = {0: 0, 1: 0, 2: 0}
x = np.arange(WINDOW_BINS) * BIN_SEC     # 窗口内的时间轴,用来算斜率

for (p, t), block in df.groupby(["Participant", "Task"]):
    signal = block.sort_values("Time")["Pupil_z"].to_numpy()
    label = block["Label"].iloc[0]

    for start in range(0, len(signal) - WINDOW_BINS + 1, STEP_BINS):
        w = signal[start:start + WINDOW_BINS]
        if np.isnan(w).any():            # 有长缺口,整段不要
            n_dropped[label] += 1
            continue
        rows.append({
            "Participant": p,
            "Task": t,
            "window_start_sec": start * BIN_SEC,
            "mean": w.mean(),
            "std": w.std(),
            "min": w.min(),
            "max": w.max(),
            "slope": np.polyfit(x, w, 1)[0],   # 每秒变化多少
            "Label": label,
        })

feat = pd.DataFrame(rows)

# --- 体检 ---
print("\n窗口总数:", len(feat))
print("\n=== 每个类别的窗口数 ===")
print(feat["Label"].map(LABEL_NAMES).value_counts())
print("\n=== 因含缺口被丢弃的窗口数 ===")
for k, v in n_dropped.items():
    print(f"{LABEL_NAMES[k]}: {v}")

per_block = feat.groupby(["Participant", "Task"]).size()
print("\n每个 block 切出的窗口数:")
print(per_block.describe())
print("一个窗口都没切出来的 block 数:", 186 - len(per_block))

feat.to_parquet("../data/features.parquet", index=False)
print("\n已保存: data/features.parquet")