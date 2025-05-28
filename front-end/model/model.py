import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os

driver_names = {
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

def make_predictions(race_data, track_data):
    file_path = os.path.join(os.path.dirname(__file__), "both_years.csv")
    df = pd.read_csv(file_path) 
    target = "average_lap_time"

    df = df[df[target] < df[target].quantile(0.99)]

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(df[[target]]).flatten()

    X = df.drop(columns=[target])
    y = y_scaled

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_train, X_valtest, y_train, y_valtest = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)

    class F1Dataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_loader = DataLoader(F1Dataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(F1Dataset(X_val, y_val), batch_size=16)
    test_loader = DataLoader(F1Dataset(X_test, y_test), batch_size=16)

    class F1Predictor(nn.Module):
        def __init__(self, input_size):
            super(F1Predictor, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.model(x)

    input_size = X.shape[1]
    model = F1Predictor(input_size)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 15
    trigger_times = 0

    for epoch in range(100):
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
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
    preds, true_vals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb))
            true_vals.append(yb)

    preds = torch.cat(preds).numpy().flatten()
    true_vals = torch.cat(true_vals).numpy().flatten()

    preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    true_vals = y_scaler.inverse_transform(true_vals.reshape(-1, 1)).flatten()

    # --------------------------- Final step, prediction: ---------------------------
    # Align columns with training data
    train_columns = df.drop(columns=[target])  # Features used during training
    train_columns = pd.get_dummies(train_columns).columns  # Columns after one-hot encoding

    results = []

    for i in range(len(race_data)):
        new_data = {
        "driver_number": race_data[i]["driver_number"],
        "year": track_data["year"],
        "lap_length": track_data["lap_length"],
        "air_temperature": track_data["air_temperature"],
        "humidity": track_data["humidity"],
        "pressure": track_data["pressure"],
        "rainfall": track_data["rainfall"],
        "track_temperature": track_data["track_temperature"],
        "wind_direction": track_data["wind_direction"],
        "wind_speed": track_data["wind_speed"],
        "quali_lap_time": race_data[i]["quali_lap_time"],
        "quali_position": race_data[i]["quali_position"],
        }

        new_df = pd.DataFrame([new_data])
        new_df = pd.get_dummies(new_df)
        new_df = new_df.reindex(columns=train_columns, fill_value=0)
        new_X = scaler.transform(new_df)
        new_tensor = torch.tensor(new_X, dtype=torch.float32)
        model.eval()
        predicted_time = 0
        with torch.no_grad():
            prediction = model(new_tensor)
            predicted_time = y_scaler.inverse_transform(prediction.numpy()).flatten()[0]

        results.append({"Driver":driver_names.get(race_data[i]["driver_number"], "Unknown Driver"), "Predicted average lap time (seconds)":predicted_time})
    sorted_results = sorted(results, key=lambda x: x["Predicted average lap time (seconds)"])
    return sorted_results, train_losses, val_losses
