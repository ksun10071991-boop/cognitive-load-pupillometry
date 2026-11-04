"""
02_preprocess.py
----------------
预处理:把原始数据变成每个 block 一条干净、等间隔的瞳孔曲线。

步骤:
1. 排除数据混乱的 block(P_18 的 Task_3 和 Task_4,两个任务时间重叠)
2. 把 Confidence < 0.8 的坏点挖空成 NaN
3. 每 50 毫秒切一个格子,格子里的好点取平均(合并重复行 + 左右眼)
4. 只插补短缺口(<= MAX_GAP_SEC),长缺口保留 NaN
5. 保存结果 + 画对比图

输入: data/colet.parquet
输出: data/colet_preprocessed.parquet
      figures/preprocess_compare_P01_Task1.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONFIDENCE_THRESHOLD = 0.8
BIN_SEC = 0.05          # 每个格子 50 毫秒
MAX_GAP_SEC = 0.25       # 暂定值:超过 0.5 秒的缺口不插补。建模前需用文献确认
EXCLUDE_BLOCKS = [("P_18", "Task_3"), ("P_18", "Task_4")]

MAX_GAP_BINS = int(round(MAX_GAP_SEC / BIN_SEC))   # 0.5 秒 = 10 个格子

# ---------- 读数据 ----------
df = pd.read_parquet("../data/colet.parquet")
print("原始行数:", len(df))

# ---------- 第 1 步:排除混乱的 block ----------
is_excluded = pd.Series(False, index=df.index)
for p, t in EXCLUDE_BLOCKS:
    is_excluded = is_excluded | ((df["Participant"] == p) & (df["Task"] == t))
df = df[~is_excluded].copy()
print("排除 P_18 两个 block 后行数:", len(df))

# ---------- 第 2 步:坏点挖空 ----------
df.loc[df["Confidence"] < CONFIDENCE_THRESHOLD, "Pupil_Diameter"] = np.nan


# ---------- 第 3 步要用的工具:把一个 block 切成 50 毫秒的格子 ----------
def resample_block(block):
    t0 = block["Timestamp"].min()
    # 每一行属于第几个格子:用 (时间 - 起点) 除以 0.05 秒,取整数
    bin_idx = ((block["Timestamp"] - t0) // BIN_SEC).astype(int)
    # 同一个格子里的好点取平均(NaN 会被自动忽略;全是 NaN 的格子结果就是 NaN)
    binned = block.groupby(bin_idx)["Pupil_Diameter"].mean()
    # 一行数据都没有的格子也要补上,标成 NaN,保证时间轴连续
    binned = binned.reindex(range(bin_idx.max() + 1))
    return pd.DataFrame({"Time": binned.index * BIN_SEC,
                         "Pupil_binned": binned.values})


# ---------- 第 4 步要用的工具:只填短缺口 ----------
def fill_short_gaps(s):
    is_nan = s.isna()
    # 给每一段连续的"缺/不缺"编号,算出每段缺口有多长
    run_id = (is_nan != is_nan.shift()).cumsum()
    run_len = is_nan.groupby(run_id).transform("sum")
    long_gap = is_nan & (run_len > MAX_GAP_BINS)
    # 先全部线性插值(只填两边都有好点的缺口),再把长缺口还原成 NaN
    filled = s.interpolate(method="linear", limit_area="inside")
    filled[long_gap] = np.nan
    return filled


# ---------- 第 3 + 4 步:一个 block 一个 block 地处理 ----------
pieces = []
for (p, t), block in df.groupby(["Participant", "Task"]):
    out = resample_block(block)
    out["Pupil"] = fill_short_gaps(out["Pupil_binned"])
    out["Participant"] = p
    out["Task"] = t
    out["NASA_RTLX"] = block["NASA_RTLX"].iloc[0]
    pieces.append(out)

df_pre = pd.concat(pieces, ignore_index=True)
df_pre = df_pre[["Participant", "Task", "Time", "Pupil_binned", "Pupil", "NASA_RTLX"]]

# ---------- 体检 ----------
print("\nblock 数:", df_pre.groupby(["Participant", "Task"]).ngroups)
print("格子总数:", len(df_pre))
print("插补前缺失格子比例:", f"{df_pre['Pupil_binned'].isna().mean():.2%}")
print("插补后缺失格子比例:", f"{df_pre['Pupil'].isna().mean():.2%}")

# ---------- 第 5 步:保存 ----------
df_pre.to_parquet("../data/colet_preprocessed.parquet", index=False)
print("\n已保存: data/colet_preprocessed.parquet")

# ---------- 对比图:P_01 / Task_1 的前 5 秒 ----------
raw = df[(df["Participant"] == "P_01") & (df["Task"] == "Task_1")].copy()
raw["Time"] = raw["Timestamp"] - raw["Timestamp"].min()
raw = raw[raw["Time"] < 5]

pre = df_pre[(df_pre["Participant"] == "P_01") & (df_pre["Task"] == "Task_1")]
pre = pre[pre["Time"] < 5]

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(raw["Time"], raw["Pupil_Diameter"], linewidth=0.5, color="gray")
axes[0].set_title("P_01 / Task_1 — raw (first 5 s)")
axes[0].set_ylabel("Pupil Diameter")
axes[0].grid(True, alpha=0.3)

axes[1].plot(pre["Time"], pre["Pupil"], linewidth=1.2, color="purple")
axes[1].set_title("P_01 / Task_1 — preprocessed: 50 ms bins, both eyes averaged")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Pupil Diameter")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../figures/preprocess_compare_P01_Task1.png", dpi=120)
print("对比图已保存: figures/preprocess_compare_P01_Task1.png")