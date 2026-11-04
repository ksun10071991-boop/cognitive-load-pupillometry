"""
02_impute.py
------------
缺失值插补:清洗后时间轴上出现空洞(眨眼坏点被删),用线性插值补齐。

输入: data/colet.parquet
输出: figures/impute_compare_P01_Task1.png  (插补前后对比)
      终端打印插补前后的数据点统计

方法: 每个 (participant, task) block 内部,线性插值。
依据: 瞳孔是平滑连续的生理信号,缺口短(眨眼仅零点几秒),
      两端之间近似直线是合理估计。插值严格限制在 block 内,
      不跨 block(不同任务负荷不同,跨块插值会污染数据)。
"""

import pandas as pd
import matplotlib.pyplot as plt

RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.8

# --- 读取 + 清洗(和 01 一致) ---
df_raw = pd.read_parquet("../data/colet.parquet")
# 保留所有行,只把低置信度点的瞳孔值标记为缺失(NaN)——留住"空位"给插值
df_clean = df_raw.copy()
mask_bad = df_clean["Confidence"] < CONFIDENCE_THRESHOLD
df_clean.loc[mask_bad, "Pupil_Diameter"] = None

# --- 插补:每个 block 内部,对 Pupil_Diameter 做线性插值 ---
# 先按 block 和时间排好序,保证插值时"前后邻居"是对的
df_imputed = df_clean.sort_values(["Participant", "Task", "Timestamp"]).copy()

# transform: 对每个 block 独立做线性插值,结果长度不变、列结构不变
# 插值严格在 block 内进行,不跨 block
df_imputed["Pupil_Diameter"] = (
    df_imputed.groupby(["Participant", "Task"])["Pupil_Diameter"]
    .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
)

print("原始行数:", len(df_raw))
print("标记为缺失的点:", mask_bad.sum(), f"({mask_bad.mean():.2%})")
print("插补后行数:", len(df_imputed))
print("插补后仍缺失的点:", df_imputed["Pupil_Diameter"].isna().sum())

# --- 对比图:清洗后(有洞) vs 插补后(补齐) ---
def get_block(data, p, t):
    return data[(data["Participant"] == p) & (data["Task"] == t)].sort_values("Timestamp")

clean_block   = get_block(df_clean,   "P_01", "Task_1")
imputed_block = get_block(df_imputed, "P_01", "Task_1")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(clean_block["Timestamp"], clean_block["Pupil_Diameter"],
             linewidth=0.5, color="green")
axes[0].set_title("P_01 / Task_1 — after cleaning (gaps remain)")
axes[0].set_ylabel("Pupil Diameter")
axes[0].grid(True, alpha=0.3)

axes[1].plot(imputed_block["Timestamp"], imputed_block["Pupil_Diameter"],
             linewidth=0.5, color="purple")
axes[1].set_title("P_01 / Task_1 — after linear interpolation")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Pupil Diameter")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../figures/impute_compare_P01_Task1.png", dpi=120)
print("对比图已保存: figures/impute_compare_P01_Task1.png")