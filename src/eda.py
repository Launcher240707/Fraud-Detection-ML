import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/creditcard.csv")

print(df.head())
print(df.shape)
print(df.info())

print(df.duplicated().sum())

print(df["Class"].value_counts())

print(df[df.duplicated(keep=False)]["Class"].value_counts())

plt.figure(figsize=(10, 6))

sns.boxplot(data=df, x="Class", y="Amount")

plt.title("Transaction Amount by Class")
plt.xlabel("Class (0: Normal, 1: Fraud)")
plt.ylabel("Amount")

plt.show()

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Amount",
    hue="Class",
    bins=50,
    kde=True,
    element="step",
    stat="density",
    common_norm=False
)

plt.xscale("log")
plt.title("Transaction Amount Distribution by Class")
plt.xlabel("Amount (log scale)")
plt.ylabel("Density")

plt.show()

plt.figure(figsize=(12, 6))

sns.histplot(
    data=df,
    x="Time",
    hue="Class",
    bins=100,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Transaction Time Distribution by Class")
plt.xlabel("Time")
plt.ylabel("Density")

plt.show()

features = [f"V{i}" for i in range(1, 29)]

plt.figure(figsize=(14, 8))

sns.boxplot(
    data=df[features + ["Class"]].melt(
        id_vars="Class",
        var_name="Feature",
        value_name="Value"
    ),
    x="Feature",
    y="Value",
    hue="Class"
)

plt.title("V1–V28 Feature Distribution by Class")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()