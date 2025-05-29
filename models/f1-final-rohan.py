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
df[targetColumn] = np.log(df[targetColumn])

X = df.drop(columns=[targetColumn])
y = df[targetColumn]

scaler = StandardScaler()
XScaled = scaler.fit_transform(X)

XTrain, XValTest, yTrain, yValTest = train_test_split(XScaled, y, test_size=0.3, random_state=42)
XVal, XTest, yVal, yTest = train_test_split(XValTest, yValTest, test_size=0.5, random_state=42)

class F1Dataset(Dataset):
    def __init__(self, features, targets):
        featuresTensor = torch.tensor(features, dtype=torch.float32)

        if isinstance(targets, pd.Series):
            targetsArray = targets.values
        else:
            targetsArray = targets

        targetsTensor = torch.tensor(targetsArray, dtype=torch.float32).unsqueeze(1)

        self.features = featuresTensor
        self.targets = targetsTensor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

trainLoader = DataLoader(F1Dataset(XTrain, yTrain), batch_size=32, shuffle=True)
valLoader = DataLoader(F1Dataset(XVal, yVal), batch_size=32)
testLoader = DataLoader(F1Dataset(XTest, yTest), batch_size=32)

class F1Predictor(nn.Module):
    def __init__(self, inputSize):
        super(F1Predictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputSize, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

inputSize = X.shape[1]
model = F1Predictor(inputSize)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

bestValLoss = float('inf')
patience = 10
triggerTimes = 0
trainLosses, valLosses = [], []

epochs = 100
for epoch in range(epochs):
    model.train()
    trainLoss = 0
    for xBatch, yBatch in trainLoader:
        optimizer.zero_grad()
        output = model(xBatch)
        loss = criterion(output, yBatch)
        loss.backward()
        optimizer.step()
        trainLoss += loss.item()
    trainLoss /= len(trainLoader)
    trainLosses.append(trainLoss)

    model.eval()
    valLoss = 0
    with torch.no_grad():
        for xBatch, yBatch in valLoader:
            output = model(xBatch)
            loss = criterion(output, yBatch)
            valLoss += loss.item()
    valLoss /= len(valLoader)
    valLosses.append(valLoss)

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
yPreds, yTrue = [], []
with torch.no_grad():
    for xBatch, yBatch in testLoader:
        yPreds.append(model(xBatch))
        yTrue.append(yBatch)

yPreds = torch.cat(yPreds).squeeze().exp().numpy()
yTrue = torch.cat(yTrue).squeeze().exp().numpy()

mse = np.mean((yPreds - yTrue) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(yPreds - yTrue))

print(f"\nTest MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

for t in [1, 2, 5, 10]:
    acc = np.mean(np.abs(yPreds - yTrue) <= t) * 100
    print(f"Accuracy within ±{t} sec: {acc:.2f}%")

plt.plot(trainLosses, label="Train Loss")
plt.plot(valLosses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(yTrue, yPreds, alpha=0.6)
plt.plot([min(yTrue), max(yTrue)], [min(yTrue), max(yTrue)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel("Actual Lap Time (s)")
plt.ylabel("Predicted Lap Time (s)")
plt.title("Predicted vs Actual Lap Times")
plt.legend()
plt.grid(True)
plt.show()
