import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
    n_jobs=-1
)

model.fit(x_train, y_train)

threshold = 0.50

joblib.dump(
    model,
    "models/random_forest_fraud.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    threshold,
    "models/threshold.pkl"
)

print("Final model saved successfully!")
print()
print("Model:")
print("models/random_forest_fraud.pkl")
print()
print("Scaler:")
print("models/scaler.pkl")
print()
print("Threshold:")
print("models/threshold.pkl")
print()
print(f"Final threshold: {threshold:.2f}")