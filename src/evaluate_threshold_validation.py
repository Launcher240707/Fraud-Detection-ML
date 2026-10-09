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

x_train_model, x_val, y_train_model, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("Training data:", x_train_model.shape)
print("Validation data:", x_val.shape)
print("Test data:", x_test.shape)

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

model.fit(x_train_model, y_train_model)

y_val_prob = model.predict_proba(x_val)[:, 1]

thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

best_threshold = 0
best_f1 = 0

print("\nValidation Threshold Evaluation:")
print("-" * 80)

for threshold in thresholds:

    y_val_pred = (y_val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val, 
        y_val_pred
    ).ravel()

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1 Score: {f1:.4f} | "
        f"FP: {fp} | "
        f"FN: {fn}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest Threshold", best_threshold)
print("Best F1 Score", round(best_f1, 4))

print("\nTraining final Random Forest model...")

scaler_final = StandardScaler()

x_train[["Time", "Amount"]] = scaler_final.fit_transform(
    x_train[["Time", "Amount"]]
)

x_test[["Time", "Amount"]] = scaler_final.transform(
    x_test[["Time", "Amount"]]
)

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

final_model.fit(x_train, y_train)

y_test_prob = final_model.predict_proba(x_test)[:, 1]

y_test_pred = (
    y_test_prob >= best_threshold
).astype(int)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test, 
    y_test_prob
)

pr_auc = average_precision_score(
    y_test,
    y_test_prob
)

cm = confusion_matrix(
    y_test, 
    y_test_pred
)

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print(f"Selected Threshold: {best_threshold:.2f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"PR AUC Score: {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)