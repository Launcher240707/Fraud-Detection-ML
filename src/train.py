import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

model = LogisticRegression(
    max_iter=1000, 
    random_state=42
)

model.fit(x_train, y_train)

print("Model training completed.")

y_pred = model.predict(x_test)

print("\nPredictions distribution:")
print(pd.Series(y_pred).value_counts())