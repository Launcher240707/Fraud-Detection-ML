import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)

df = pd.read_csv("data/creditcard.csv")

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

x_train[["Time", "Amount"]] = scaler.fit_transform(
    x_train[["Time", "Amount"]]
)

x_test[["Time", "Amount"]] = scaler.transform(
    x_test[["Time", "Amount"]]
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

model.fit(x_train, y_train)

y_prob = model.predict_proba(x_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_test, 
    y_prob
)

pr_auc = average_precision_score(
    y_test, 
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f"Random Forest (AP = {pr_auc:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - Random Forest Fraud Detection"
)

plt.legend()
plt.grid()

plt.savefig(
    "results/figures/precision_recall_curve.png", 
    dpi=300
)

plt.show()

print(f"PR AUC score: {pr_auc:.4f}")

print("Precision-Recall curve saved to:")
print("results/figures/precision_recall_curve.png")