import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./data/f1_cleaned_with_year.csv")
targetColumn = "average_lap_time"

df = df[df[targetColumn] < df[targetColumn].quantile(0.99)]

yScaler = StandardScaler()
yScaled = yScaler.fit_transform(df[[targetColumn]]).flatten()

X = df.drop(columns=[targetColumn])
y = yScaled

scaler = StandardScaler()
XScaled = scaler.fit_transform(X)

XTrain, XValTest, yTrain, yValTest = train_test_split(XScaled, y, test_size=0.3, random_state=42)
XVal, XTest, yVal, yTest = train_test_split(XValTest, yValTest, test_size=0.5, random_state=42)

class F1Dataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

trainLoader = DataLoader(F1Dataset(XTrain, yTrain), batch_size=15, shuffle=True)
valLoader = DataLoader(F1Dataset(XVal, yVal), batch_size=15)
testLoader = DataLoader(F1Dataset(XTest, yTest), batch_size=15)

class F1Predictor(nn.Module):
    def __init__(self, inputSize):
        super(F1Predictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputSize, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

inputSize = X.shape[1]
model = F1Predictor(inputSize)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

trainLosses, valLosses = [], []
bestValLoss = float('inf')
patience = 15
triggerTimes = 0

epochs = 100
for epoch in range(epochs):
    model.train()
    totalTrainLoss = 0
    for xb, yb in trainLoader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        totalTrainLoss += loss.item()
    trainLoss = totalTrainLoss / len(trainLoader)

    model.eval()
    totalValLoss = 0
    with torch.no_grad():
        for xb, yb in valLoader:
            pred = model(xb)
            loss = criterion(pred, yb)
            totalValLoss += loss.item()
    valLoss = totalValLoss / len(valLoader)

    trainLosses.append(trainLoss)
    valLosses.append(valLoss)
    scheduler.step(valLoss)

    print(f"Epoch {epoch+1}: Train Loss = {trainLoss:.4f}, Validation Loss = {valLoss:.4f}")

    if valLoss < bestValLoss:
        bestValLoss = valLoss
        triggerTimes = 0
        bestModelState = model.state_dict()
    else:
        triggerTimes += 1
        if triggerTimes >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(bestModelState)

model.eval()
preds, trueVals = [], []
with torch.no_grad():
    for xb, yb in testLoader:
        preds.append(model(xb))
        trueVals.append(yb)

preds = torch.cat(preds).numpy().flatten()
trueVals = torch.cat(trueVals).numpy().flatten()

preds = yScaler.inverse_transform(preds.reshape(-1, 1)).flatten()
trueVals = yScaler.inverse_transform(trueVals.reshape(-1, 1)).flatten()

mse = np.mean((preds - trueVals) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(preds - trueVals))

print(f"\nTest MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

for t in [1, 2, 5, 10]:
    print(f"Accuracy within ±{t}s: {np.mean(np.abs(preds - trueVals) <= t) * 100:.2f}%")

plt.plot(trainLosses, label="Train Loss")
plt.plot(valLosses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(trueVals, preds, alpha=0.6)
plt.plot([min(trueVals), max(trueVals)], [min(trueVals), max(trueVals)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel("Actual Lap Time (s)")
plt.ylabel("Predicted Lap Time (s)")
plt.title("Predicted vs Actual Lap Times")
plt.legend()
plt.grid(True)
plt.show()
