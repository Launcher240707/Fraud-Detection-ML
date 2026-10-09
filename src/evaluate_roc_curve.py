import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
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

fpr, tpr, thresholds = roc_curve(
    y_test, 
    y_prob
)

roc_auc = roc_auc_score(
    y_test, 
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1], 
    [0, 1], 
    linestyle="--", 
    label="Random Guessing"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "results/figures/roc_curve.png",
    dpi=300
)

plt.show()

print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"ROC curve saved to:")
print("results/figures/roc_curve.png")