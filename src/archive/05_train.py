"""
05_train.py
-----------
按受试者划分训练集/测试集(约 8:2),训练模型,与 baseline 比较。

划分: 同一个受试者的所有窗口只能出现在训练集或测试集之一(防止泄露)。
Baseline: 永远预测出现最多的类别(DummyClassifier),在同一个测试集上评估。
模型: Logistic Regression、Random Forest,均使用 class_weight="balanced" 应对类别不均衡。
指标: Accuracy、Macro-F1(三类平均,不被多数类掩盖)、分类报告、混淆矩阵。

输入: data/features.parquet
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

RANDOM_SEED = 42
FEATURES = ["mean", "std", "min", "max", "slope"]
LABEL_NAMES = ["low", "medium", "high"]

feat = pd.read_parquet("../data/features.parquet")
X = feat[FEATURES]
y = feat["Label"]
groups = feat["Participant"]

# --- 按人划分:约 80% 的人教模型,20% 的人考试 ---
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_people = set(groups.iloc[train_idx])
test_people = set(groups.iloc[test_idx])
print("训练组人数:", len(train_people), "| 考试组人数:", len(test_people))
print("两组重叠的人数(必须是 0):", len(train_people & test_people))
print("\n考试组各类窗口数:")
print(y_test.map(dict(enumerate(LABEL_NAMES))).value_counts())

# --- 三个"考生" ---
models = {
    "Baseline (always most frequent)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000)),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED),
}

for name, model in models.items():
    model.fit(X_train, y_train)          # 只用训练组学习
    pred = model.predict(X_test)         # 在考试组上考试

    print(f"\n========== {name} ==========")
    print("Accuracy:", round(accuracy_score(y_test, pred), 3))
    print("Macro-F1:", round(f1_score(y_test, pred, average="macro"), 3))
    print(classification_report(y_test, pred, labels=[0, 1, 2],
                                target_names=LABEL_NAMES, zero_division=0))
    print("混淆矩阵(行 = 真实,列 = 预测,顺序 low / medium / high):")
    print(confusion_matrix(y_test, pred, labels=[0, 1, 2]))