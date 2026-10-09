import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE

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

smote = SMOTE(random_state=42)

x_train_resampled, y_train_resampled = smote.fit_resample(
    x_train, 
    y_train
)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_resampled.value_counts())

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

model.fit(x_train_resampled, y_train_resampled)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"PR AUC Score: {pr_auc:.4f}")