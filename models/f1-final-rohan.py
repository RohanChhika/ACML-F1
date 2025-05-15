import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("./data/f1_cleaned_with_year.csv")
target = "average_lap_time"

# Clip outliers
df = df[df[target] < df[target].quantile(0.99)]

# Log transform target
df[target] = np.log(df[target])

# Features and target
X = df.drop(columns=[target])
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Val/Test split (70/15/15)
X_train, X_valtest, y_train, y_valtest = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)

# Dataset class
class F1Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Dataloaders
train_loader = DataLoader(F1Dataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(F1Dataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(F1Dataset(X_test, y_test), batch_size=32)

# Model
class F1Predictor(nn.Module):
    def __init__(self, input_size):
        super(F1Predictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Training config
input_size = X.shape[1]
model = F1Predictor(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Early stopping settings
best_val_loss = float('inf')
patience = 10
trigger_times = 0
train_losses, val_losses = [], []

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        best_model_state = model.state_dict()
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(best_model_state)

# Evaluate
model.eval()
y_preds, y_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_preds.append(model(X_batch))
        y_true.append(y_batch)

y_preds = torch.cat(y_preds).squeeze().exp().numpy()
y_true = torch.cat(y_true).squeeze().exp().numpy()

# Metrics
mse = np.mean((y_preds - y_true) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_preds - y_true))

print(f"\nTest MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Accuracy buckets
def accuracy_within(y_true, y_pred, tolerance):
    return np.mean(np.abs(y_pred - y_true) <= tolerance) * 100

for t in [1, 2, 5, 10]:
    acc = accuracy_within(y_true, y_preds, t)
    print(f"Accuracy within ±{t} sec: {acc:.2f}%")

# Loss plot
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.show()


# Test Results Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_preds, alpha=0.6)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel("Actual Lap Time (s)")
plt.ylabel("Predicted Lap Time (s)")
plt.title("Predicted vs Actual Lap Times")
plt.legend()
plt.grid(True)
plt.show()

