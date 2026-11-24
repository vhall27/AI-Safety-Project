import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/wdbc.data")

train, temp = train_test_split(
    df, test_size=0.30, random_state=42,
    stratify=df["Diagnosis"]
)

val, test = train_test_split(
    temp, test_size=0.50, random_state=42,
    stratify=temp["Diagnosis"]
)

train.to_csv("data/train.csv", index=False)
val.to_csv("data/validation.csv", index=False)
test.to_csv("data/test.csv", index=False)

# Loading in the three datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")
test = pd.read_csv("data/test.csv")

# Removing patient ID numbers and turning diagnosis to binary label
X_train = train.drop(columns=["ID", "Diagnosis"])
X_val = val.drop(columns=["ID", "Diagnosis"])
X_test = test.drop(columns=["ID", "Diagnosis"])

y_train = train["Diagnosis"].map({"M": 1, "B": 0})
y_val = val["Diagnosis"].map({"M": 1, "B": 0})
y_test = test["Diagnosis"].map({"M": 1, "B": 0})

scaler = StandardScaler()

# Calculate training means and standard deviations,
# then standardize the training data.
X_train_scaled = scaler.fit_transform(X_train)

# Standardize using the existing training statistics.
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Training size:", len(X_train))
print("Validation size:", len(X_val))
print("Testing size:", len(X_test))

df = pd.read_csv("data/wdbc.data")
print(df.shape)
print(df["Diagnosis"].value_counts())