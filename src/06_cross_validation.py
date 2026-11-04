"""
06_cross_validation.py
----------------------
5 折按受试者分层交叉验证:每个受试者都恰好当一次测试集。
报告每个模型各指标的 均值 ± 标准差,以及 5 折合并的混淆矩阵。

划分: StratifiedGroupKFold —— 同一受试者不跨折(Group),各折类别比例尽量均衡(Stratified)。
每一折都重新创建模型,保证上一折学到的东西不会带到下一折。

输入: data/features.parquet
输出: results/cv_results.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RANDOM_SEED = 42
N_FOLDS = 5
FEATURES = ["mean", "std", "min", "max", "slope"]
LABELS = [0, 1, 2]
LABEL_NAMES = ["low", "medium", "high"]

feat = pd.read_parquet("../data/features.parquet")
X = feat[FEATURES]
y = feat["Label"]
groups = feat["Participant"]


def make_models():
    """每次调用都返回一组全新的、没学过任何东西的模型。"""
    return {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED),
    }


cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

results = []
total_cm = {name: np.zeros((3, 3), dtype=int) for name in make_models()}

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    train_people = set(groups.iloc[train_idx])
    test_people = set(groups.iloc[test_idx])
    assert len(train_people & test_people) == 0, "出现了跨组的受试者!"

    y_true = y.iloc[test_idx]
    print(f"第 {fold} 折: 考试组 {len(test_people)} 人, "
          f"窗口数 low/medium/high = "
          f"{(y_true == 0).sum()}/{(y_true == 1).sum()}/{(y_true == 2).sum()}")

    for name, model in make_models().items():
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        f1_each = f1_score(y_true, pred, labels=LABELS, average=None, zero_division=0)
        results.append({
            "fold": fold,
            "model": name,
            "accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, labels=LABELS,
                                 average="macro", zero_division=0),
            "f1_low": f1_each[0],
            "f1_medium": f1_each[1],
            "f1_high": f1_each[2],
        })
        total_cm[name] += confusion_matrix(y_true, pred, labels=LABELS)

res = pd.DataFrame(results)

# --- 汇总:每个模型每个指标的 均值 ± 标准差 ---
metrics = ["accuracy", "macro_f1", "f1_low", "f1_medium", "f1_high"]
for name in make_models():
    sub = res[res["model"] == name]
    print(f"\n========== {name} ==========")
    for m in metrics:
        print(f"{m:10s}: {sub[m].mean():.3f} ± {sub[m].std():.3f}")
    print("5 折合并混淆矩阵(行 = 真实,列 = 预测,顺序 low / medium / high):")
    print(total_cm[name])

os.makedirs("../results", exist_ok=True)
res.to_csv("../results/cv_results.csv", index=False)
print("\n每一折的详细结果已保存: results/cv_results.csv")