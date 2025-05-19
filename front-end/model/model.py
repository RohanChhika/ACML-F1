import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def make_predictions(race_data, track_data):
    # Load cleaned dataset
    df = pd.read_csv("../model/both_years.csv") 
    target = "average_lap_time"

    #Preprocessing: average_lap_time

    # Clip outliers
    df = df[df[target] < df[target].quantile(0.99)]

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(df[[target]]).flatten()

    # Features and target
    X = df.drop(columns=[target])
    y = y_scaled

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Data Split

    # Train/Val/Test split (70/15/15)
    X_train, X_valtest, y_train, y_valtest = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)


    # Dataset class
    class F1Dataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # Dataloaders
    train_loader = DataLoader(F1Dataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(F1Dataset(X_val, y_val), batch_size=16)
    test_loader = DataLoader(F1Dataset(X_test, y_test), batch_size=16)

    # Model
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

    # Training config
    input_size = X.shape[1]
    model = F1Predictor(input_size)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Early stopping settings
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 15
    trigger_times = 0

    # Training loop
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
    preds, true_vals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb))
            true_vals.append(yb)

    preds = torch.cat(preds).numpy().flatten()
    true_vals = torch.cat(true_vals).numpy().flatten()

    # Inverse scale predictions
    preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    true_vals = y_scaler.inverse_transform(true_vals.reshape(-1, 1)).flatten()

    # Metrics
    mse = np.mean((preds - true_vals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - true_vals))

    print(f"\nTest MSE: {mse:.4f}")

        #<5 is decent, <2 is good, <1 is great
    print(f"Test RMSE: {rmse:.4f}")

    print(f"Test MAE: {mae:.4f}")

    # Accuracy buckets
    def accuracy_within(y_true, y_pred, tolerance):
        return np.mean(np.abs(y_pred - y_true) <= tolerance) * 100

    for t in [1, 2, 5, 10]:
        print(f"Accuracy within ±{t}s: {accuracy_within(true_vals, preds, t):.2f}%")


    # --------------------------- Final step, prediction: ---------------------------
    # Align columns with training data
    train_columns = df.drop(columns=[target])  # Features used during training
    train_columns = pd.get_dummies(train_columns).columns  # Columns after one-hot encoding

    # Reindex the new data to match training feature order, fill missing with 0
    new_df = new_df.reindex(columns=train_columns, fill_value=0)

    # Scale
    new_X = scaler.transform(new_df)

    # Predict
    new_tensor = torch.tensor(new_X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(new_tensor)
        predicted_time = y_scaler.inverse_transform(prediction.numpy()).flatten()[0]

    print("Predicted average lap time:", predicted_time)

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

        # Convert to DataFrame
        new_df = pd.DataFrame([new_data])

        # Encode categoricals like during training
        new_df = pd.get_dummies(new_df)

        # Reindex the new data to match training feature order, fill missing with 0
        new_df = new_df.reindex(columns=train_columns, fill_value=0)

        # Scale
        new_X = scaler.transform(new_df)

        # Predict
        new_tensor = torch.tensor(new_X, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            prediction = model(new_tensor)
            predicted_time = y_scaler.inverse_transform(prediction.numpy()).flatten()[0]

        print("Predicted average lap time for driver number",i,":", predicted_time)

# # Loss plot
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Loss per Epoch")
# plt.legend()
# plt.show()

race_data = [
    {
        "driver_number": 4,
        "quali_lap_time": 230.5537333333333,
        "quali_position": 1
    },
    {
        "driver_number": 55,
        "quali_lap_time": 233.15906666666666,
        "quali_position": 3
    },
    {
        "driver_number": 16,
        "quali_lap_time": 209.16309090909093,
        "quali_position": 14
    },
    {
        "driver_number": 44,
        "quali_lap_time": 166.47116666666665,
        "quali_position": 18
    },
    {
        "driver_number": 63,
        "quali_lap_time": 208.1088235294118,
        "quali_position": 7
    },
    {
        "driver_number": 1,
        "quali_lap_time": 263.95015384615385,
        "quali_position": 5
    },
    {
        "driver_number": 10,
        "quali_lap_time": 204.6325,
        "quali_position": 6
    },
    {
        "driver_number": 27,
        "quali_lap_time": 210.465,
        "quali_position": 4
    },
    {
        "driver_number": 14,
        "quali_lap_time": 211.74764705882353,
        "quali_position": 8
    },
    {
        "driver_number": 81,
        "quali_lap_time": 227.6352666666667,
        "quali_position": 2
    },
    {
        "driver_number": 23,
        "quali_lap_time": 123.84540000000001,
        "quali_position": 16
    },
    {
        "driver_number": 22,
        "quali_lap_time": 214.64072727272725,
        "quali_position": 11
    },
    {
        "driver_number": 24,
        "quali_lap_time": 146.944,
        "quali_position": 17
    },
    {
        "driver_number": 18,
        "quali_lap_time": 194.0543076923077,
        "quali_position": 13
    },
    {
        "driver_number": 61,
        "quali_lap_time": 159.0538,
        "quali_position": 20
    },
    {
        "driver_number": 20,
        "quali_lap_time": 216.76800000000003,
        "quali_position": 15
    },
    {
        "driver_number": 30,
        "quali_lap_time": 211.7084545454546,
        "quali_position": 12
    },
    {
        "driver_number": 77,
        "quali_lap_time": 253.5219230769231,
        "quali_position": 9
    },
    {
        "driver_number": 43,
        "quali_lap_time": 121.67840000000001,
        "quali_position": 19
    }
]

track_data = {
    "year": 2024,
    "lap_length": 5.554,
        "air_temperature": 25.8,
        "humidity": 60.0,
        "pressure": 1018.0,
        "rainfall": 0,
        "track_temperature": 29.1,
        "wind_direction": 116,
        "wind_speed": 1.2,
}