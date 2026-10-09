import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
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

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"PR AUC Score: {pr_auc:.4f}")

print("\nThreshold Evaluation:")
print("-" * 80)

thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_test, 
        y_pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_test, 
        y_pred_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y_test, 
        y_pred_threshold,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, 
        y_pred_threshold
    ).ravel()

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1 Score: {f1:.4f} | "
        f"FP: {fp} | "
        f"FN: {fn}"
    )