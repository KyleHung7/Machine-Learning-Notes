"""
Titanic Classification with PyTorch (Local VSCode Version)
Fully cleaned, works with train.csv/test.csv from Kaggle
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

print("Train head:")
print(train.head(5))

# ------------------------------------------------------------
# 2. Data Cleaning / Feature Creation
# ------------------------------------------------------------
# Fill missing values
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
train["Family_Size"] = train["Parch"] + train["SibSp"] + 1

# Test data
test["Age"] = test["Age"].fillna(train["Age"].median())
test["Embarked"] = test["Embarked"].fillna("S")
test["Fare"] = test["Fare"].fillna(train["Fare"].median())
test["Family_Size"] = test["Parch"] + test["SibSp"] + 1

# ------------------------------------------------------------
# 3. Family Size Category
# ------------------------------------------------------------
def family_size_category(x):
    if 2 <= x <= 4:
        return "High_Survival_Rates"
    else:
        return "Low_Survival_Rates"

train["Family_Size_Category"] = train["Family_Size"].map(family_size_category)
test["Family_Size_Category"] = test["Family_Size"].map(family_size_category)

# ------------------------------------------------------------
# 4. Remove unusable columns
# ------------------------------------------------------------
drop_cols = ["Name", "Ticket", "Cabin"]
train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

# ------------------------------------------------------------
# 5. One-hot encoding categorical features
# ------------------------------------------------------------
def one_hot_encode(df):
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df["Embarked"] = df["Embarked"].astype("category").cat.codes
    df["Family_Size_Category"] = df["Family_Size_Category"].astype("category").cat.codes
    return df

train = one_hot_encode(train)
test = one_hot_encode(test)

# ------------------------------------------------------------
# 6. Train / Validation Split
# ------------------------------------------------------------
x_train_full = train.drop(columns=["Survived", "PassengerId"])
y_train_full = train["Survived"]

train_x, val_x, train_y, val_y = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=777
)

# ------------------------------------------------------------
# 7. Convert to PyTorch Dataset
# ------------------------------------------------------------
class TitanicDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(np.array(x), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

batch_size = 100
train_set = TitanicDataset(train_x, train_y)
val_set = TitanicDataset(val_x, val_y)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ------------------------------------------------------------
# 8. Define Neural Network
# ------------------------------------------------------------
class NN(nn.Module):
    def __init__(self, input_dim):
        super(NN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.layer3 = nn.Linear(16, 2)  # binary classification → 2 outputs

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = NN(input_dim=train_x.shape[1])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------------------
# 9. Training Loop
# ------------------------------------------------------------
epochs = 50
print("Training on device:", device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f}")

print("Training finished!")

# ------------------------------------------------------------
# 10. Validation
# ------------------------------------------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = correct / total * 100
print(f"Validation Accuracy: {accuracy:.2f}%")

# ------------------------------------------------------------
# 11. Optional: Predict on test.csv
# ------------------------------------------------------------
test_x = test.drop(columns=["PassengerId"])
test_tensor = torch.tensor(np.array(test_x), dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    test_outputs = model(test_tensor)
    _, test_pred = torch.max(test_outputs, 1)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_pred.cpu().numpy()
})
submission.to_csv("submission.csv", index=False)
print("Saved test predictions to submission.csv")
