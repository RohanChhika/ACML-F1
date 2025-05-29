import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

driverNames = {
    1: 'Max VERSTAPPEN', 2: 'Logan SARGEANT', 4: 'Lando NORRIS',
    10: 'Pierre GASLY', 11: 'Sergio PEREZ', 14: 'Fernando ALONSO',
    16: 'Charles LECLERC', 20: 'Kevin MAGNUSSEN', 21: 'Maxwell ESTERSON',
    22: 'Yuki TSUNODA', 24: 'ZHOU Guanyu', 27: 'Nico HULKENBERG',
    31: 'Esteban OCON', 44: 'Lewis HAMILTON', 55: 'Carlos SAINZ',
    63: 'George RUSSELL', 81: 'Oscar PIASTRI', 23: 'Alexander ALBON',
    34: 'Felipe DRUGOVICH', 77: 'Valtteri BOTTAS', 18: 'Lance STROLL',
    3: 'Daniel RICCIARDO', 5: 'Gabriel BORTOLETO', 6: 'Isack HADJAR',
    7: 'Jack DOOHAN', 8: 'Gregoire SAUCY', 9: 'Nikola TSOLOV',
    12: 'Kimi ANTONELLI', 15: 'Gabriele MINI', 17: 'Caio COLLET',
    19: 'Tommy SMITH', 25: 'Hugh BARTER', 26: 'Nikita BEDRIN',
    28: 'Ryo HIRAKAWA', 29: "Patricio O'WARD", 30: 'Liam LAWSON',
    39: 'Arthur LECLERC', 40: 'Ayumu IWASA', 41: 'Isack HADJAR',
    42: 'Frederik VESTI', 50: 'Ryo HIRAKAWA', 61: 'Jack DOOHAN',
    98: 'Theo POURCHAIRE', 36: 'Jake DENNIS', 37: 'Ayumu IWASA',
    45: 'Franco COLAPINTO', 38: 'Dino BEGANOVIC', 97: 'Robert SHWARTZMAN',
    43: 'Franco COLAPINTO', 46: 'Luke BROWNING', 87: 'Oliver BEARMAN',
    62: 'Ryo HIRAKAWA', 72: 'Frederik VESTI'
}

def make_predictions(raceData, trackData):
    filePath = os.path.join(os.path.dirname(__file__), "both_years.csv")
    df = pd.read_csv(filePath) 
    target = "average_lap_time"

    df = df[df[target] < df[target].quantile(0.99)]

    yScalar = StandardScaler()
    yScaled = yScalar.fit_transform(df[[target]]).flatten()

    X = df.drop(columns=[target])
    y = yScaled

    scaler = StandardScaler()
    xScaled = scaler.fit_transform(X)


    X_train, X_valtest, y_train, y_valtest = train_test_split(xScaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)

    class F1Dataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    trainLoader = DataLoader(F1Dataset(X_train, y_train), batch_size=16, shuffle=True)
    valLoader = DataLoader(F1Dataset(X_val, y_val), batch_size=16)
    testLoader = DataLoader(F1Dataset(X_test, y_test), batch_size=16)

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
    best_val_loss = float('inf')
    patience = 15
    trigger_times = 0

    for epoch in range(100):
        model.train()
        totalTrainLoss = 0
        for xb, yb in trainLoader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            totalTrainLoss += loss.item()
        train_loss = totalTrainLoss / len(trainLoader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in valLoader:
                pred = model(xb)
                loss = criterion(pred, yb)
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(valLoader)

        trainLosses.append(train_loss)
        valLosses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            best_model_state = model.state_dict()
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

    model.load_state_dict(best_model_state)

    model.eval()
    preds, trueVals = [], []
    with torch.no_grad():
        for xb, yb in testLoader:
            preds.append(model(xb))
            trueVals.append(yb)

    preds = torch.cat(preds).numpy().flatten()
    trueVals = torch.cat(trueVals).numpy().flatten()

    preds = yScalar.inverse_transform(preds.reshape(-1, 1)).flatten()
    trueVals = yScalar.inverse_transform(trueVals.reshape(-1, 1)).flatten()

    trainCols = df.drop(columns=[target])  
    trainCols = pd.get_dummies(trainCols).columns  

    results = []

    for i in range(len(raceData)):
        newData = {
        "driver_number": raceData[i]["driver_number"],
        "year": trackData["year"],
        "lap_length": trackData["lap_length"],
        "air_temperature": trackData["air_temperature"],
        "humidity": trackData["humidity"],
        "pressure": trackData["pressure"],
        "rainfall": trackData["rainfall"],
        "track_temperature": trackData["track_temperature"],
        "wind_direction": trackData["wind_direction"],
        "wind_speed": trackData["wind_speed"],
        "quali_lap_time": raceData[i]["quali_lap_time"],
        "quali_position": raceData[i]["quali_position"],
        }

        newDf = pd.DataFrame([newData])
        newDf = pd.get_dummies(newDf)
        newDf = newDf.reindex(columns=trainCols, fill_value=0)
        newX = scaler.transform(newDf)
        newTensor = torch.tensor(newX, dtype=torch.float32)
        model.eval()
        predictedTime = 0
        with torch.no_grad():
            prediction = model(newTensor)
            predictedTime = yScalar.inverse_transform(prediction.numpy()).flatten()[0]

        results.append({"Driver":driverNames.get(raceData[i]["driver_number"], "Unknown Driver"), "Predicted average lap time (seconds)":predictedTime})
    sorted_results = sorted(results, key=lambda x: x["Predicted average lap time (seconds)"])
    return sorted_results, trainLosses, valLosses
