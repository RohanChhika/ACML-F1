import os
import sys
import pandas as pd
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualisations.plotter import Plotter

losses = []

df = pd.read_csv("../data/both_years.csv")

targetColumn = "average_lap_time"
categoricalColumns = ["year", "driver_number", "quali_position"]

df = pd.get_dummies(df, columns=categoricalColumns)

X = df.drop(columns=[targetColumn])
y = df[targetColumn]

scaler = StandardScaler()
X = scaler.fit_transform(X)

XTrain, XTest, yTrain, yTest = train_test_split(X, y.values, test_size=0.2, random_state=42)

class F1Dataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

trainDataset = F1Dataset(XTrain, yTrain)
testDataset = F1Dataset(XTest, yTest)

trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=32)

class F1Predictor(nn.Module):
    def __init__(self, inputSize):
        super(F1Predictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputSize, 64),
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

epochs = 100
for epoch in range(epochs):
    model.train()
    runningLoss = 0.0

    for features, labels in trainLoader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
    avgLoss = runningLoss / len(trainLoader)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avgLoss:.4f}")

    losses.append(avgLoss)

model.eval()
preds, trueVals = [], []
maeTotal = 0

with torch.no_grad():
    totalLoss = 0
    for features, labels in testLoader:
        outputs = model(features)
        preds.append(outputs)
        trueVals.append(labels)

        loss = criterion(outputs, labels)
        totalLoss += loss.item()

        maeTotal += torch.mean(torch.abs(outputs - labels)).item()

    avgTestLoss = totalLoss / len(testLoader)
    avgMAE = maeTotal / len(testLoader) 

    print(f"Test MSE: {avgTestLoss:.4f}")
    rmse = np.sqrt(avgTestLoss)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {avgMAE:.4f}")

preds = torch.cat(preds).squeeze().numpy()
trueVals = torch.cat(trueVals).squeeze().numpy()

for t in [1, 2, 5, 10]:
    print(f"Accuracy within ±{t}s: {np.mean(np.abs(preds - trueVals) <= t) * 100:.2f}%")


plt.figure(figsize=(10, 6))
plt.scatter(trueVals, preds, alpha=0.6)
plt.plot([min(trueVals), max(trueVals)], [min(trueVals), max(trueVals)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel("Actual Lap Time (s)")
plt.ylabel("Predicted Lap Time (s)")
plt.title("Predicted vs Actual Lap Times")
plt.legend()
plt.grid(True)
plt.show()

plotter = Plotter()
plotter.plot_loss(losses)

