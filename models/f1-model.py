import pandas as pd
import torch
import sys
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualisations.plotter import Plotter
#meghan initial with 2024
losses=[]

# 1. Load dataset
df = pd.read_csv("./data/data-with-lap-length/2024.csv") 

# 2. Define target
target = "average_lap_time"

# 3. One-hot encode categorical variables
categorical_cols = ["session_type", "year", "driver_number", "driver_name", "position"]
df = pd.get_dummies(df, columns=categorical_cols)

# 4. Separate features and target
X = df.drop(columns=[target])
y = df[target]

# 5. Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2, random_state=42)

# 7. PyTorch Dataset
class F1Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = F1Dataset(X_train, y_train)
test_dataset = F1Dataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 8. Define model
class F1Predictor(nn.Module):
    def __init__(self, input_size):
        super(F1Predictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

model = F1Predictor(X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 9. Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    losses.append(running_loss/len(train_loader))

# 10. Evaluation
model.eval()
with torch.no_grad():
    total_loss = 0
    for features, labels in test_loader:
        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    print(f"Test MSE: {total_loss/len(test_loader):.4f}")

plotter = Plotter()
plotter.plot_loss(losses)
