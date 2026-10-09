import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    x_train[["Time", "Amount"]])

x_test[["Time", "Amount"]] = scaler.transform(
    x_test[["Time", "Amount"]])

print("Training data:", 
      x_train.shape)
print("Testing data:", 
      x_test.shape)

print("\nTraining class distribution:", 
      y_train.value_counts())

print("\nTesting class distribution:", 
      y_test.value_counts())

print("\nScaled training data:")
print(x_train[["Time", "Amount"]].head())