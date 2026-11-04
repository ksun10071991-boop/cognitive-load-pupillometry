"""
01_clean.py
-----------
数据清洗:去除低置信度的脏点(眨眼/视线偏移伪迹)。

输入: data/colet.parquet  (原始 COLET 眼动数据)
输出: figures/clean_compare_P01_Task1.png  (清洗前后对比图)
      终端打印数据质量报告

清洗策略: 丢弃 Confidence < 0.8 的采样点。
依据: 项目要求;已验证这些点对应眨眼伪迹(Confidence 中位数为 0)。
"""

import pandas as pd
import matplotlib.pyplot as plt

# 固定随机种子,保证可复现(本脚本暂未用到随机性,作为习惯保留)
RANDOM_SEED = 42

# 置信度阈值(项目要求规定)
CONFIDENCE_THRESHOLD = 0.8

# --- 读取原始数据 ---
# 脚本在 src/ 里运行,数据在上一层的 data/ 里,所以用 ../
df_raw = pd.read_parquet("../data/colet.parquet")
print("原始数据行数:", len(df_raw))

# --- 清洗:只保留置信度达标的点 ---
df_clean = df_raw[df_raw["Confidence"] >= CONFIDENCE_THRESHOLD].copy()

# --- 数据质量报告 ---
n_removed = len(df_raw) - len(df_clean)
pct_removed = n_removed / len(df_raw)
print("清洗后行数:", len(df_clean))
print("被丢弃行数:", n_removed)
print("丢弃比例: {:.2%}".format(pct_removed))

# --- 清洗前后对比图(以 P_01 / Task_1 为例) ---
def get_block(data, participant, task):
    """从数据里取出某个受试者某个任务的片段。"""
    return data[(data["Participant"] == participant) & (data["Task"] == task)]

before = get_block(df_raw,   "P_01", "Task_1")
after  = get_block(df_clean, "P_01", "Task_1")

# 上下两张子图:上=清洗前,下=清洗后
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(before["Timestamp"], before["Pupil_Diameter"], linewidth=0.5)
axes[0].set_title("P_01 / Task_1 — before cleaning")
axes[0].set_ylabel("Pupil Diameter")
axes[0].grid(True, alpha=0.3)

axes[1].plot(after["Timestamp"], after["Pupil_Diameter"], linewidth=0.5, color="green")
axes[1].set_title("P_01 / Task_1 — after cleaning (Confidence >= 0.8)")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Pupil Diameter")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../figures/clean_compare_P01_Task1.png", dpi=120)
print("对比图已保存: figures/clean_compare_P01_Task1.png")