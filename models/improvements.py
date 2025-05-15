import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# Load cleaned dataset
# df = pd.read_csv("./data/both_years.csv") 
df = pd.read_csv("./data/f1_cleaned_with_year.csv")
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
train_loader = DataLoader(F1Dataset(X_train, y_train), batch_size=15, shuffle=True)
val_loader = DataLoader(F1Dataset(X_val, y_val), batch_size=15)
test_loader = DataLoader(F1Dataset(X_test, y_test), batch_size=15)

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
# DRIVER_NUMBERS = [
#     1, 2, 4, 10, 11, 14, 16, 20, 21, 22, 24, 27, 31, 44, 55, 63, 81,
#     23, 34, 77, 18, 3, 5, 41, 61, 8, 9, 12, 15, 17, 19, 25, 26, 62, 29,
#     30, 39, 37, 72, 98, 36, 43, 38, 97, 46, 87
# ]

# for i in range(len(DRIVER_NUMBERS)):
#     new_data = {
#         "year": 2024,
#         "lap_length": 5.554,
#         "driver_number": DRIVER_NUMBERS[i],
#         "air_temperature":25.8,
#         "humidity":60.0,
#         "pressure":1018.0,
#         "rainfall":0,
#         "track_temperature":29.1,
#         "wind_direction":116,
#         "wind_speed":1.2,
#         "quali_position":7,
#         "quali_lap_time":208.1088235294118,
#     }


#     if DRIVER_NUMBERS[i] != 63:
#         row = df[
#             (df['year'] == new_data['year']) &
#             (df['lap_length'].round(3) == round(new_data["lap_length"], 3)) &
#             (df['driver_number'] == DRIVER_NUMBERS[i])
#         ]

#         if not row.empty:
#             new_data["quali_position"] = row.iloc[0]['quali_position']
#             new_data["quali_lap_time"] = row.iloc[0]['quali_lap_time']
#         else:
#             print(f"Skipping driver {DRIVER_NUMBERS[i]} - no data found")
#             continue
#     else:
#         new_data["quali_position"] = 7
#         new_data["quali_lap_time"] = 208.1088235294118
#     # new_data = {
#     #     "year": 2025,
#     #     "lap_length": 5.412,
#     #     "driver_number": 1,
#     #     "air_temperature":25.9,
#     #     "humidity":65.0,
#     #     "pressure":1011.9,
#     #     "rainfall":0,
#     #     "track_temperature":34.9,
#     #     "wind_direction":336,
#     #     "wind_speed":1.1,
#     #     "quali_position":1,
#     #     "quali_lap_time":127.186
#     # }

#     # Convert to DataFrame
#     new_df = pd.DataFrame([new_data])

#     # Encode categoricals like during training
#     new_df = pd.get_dummies(new_df)

#     # Align columns with training data
#     train_columns = df.drop(columns=[target])  # Features used during training
#     train_columns = pd.get_dummies(train_columns).columns  # Columns after one-hot encoding

#     # Reindex the new data to match training feature order, fill missing with 0
#     new_df = new_df.reindex(columns=train_columns, fill_value=0)

#     # Scale
#     new_X = scaler.transform(new_df)

#     # Predict
#     new_tensor = torch.tensor(new_X, dtype=torch.float32)
#     model.eval()
#     with torch.no_grad():
#         prediction = model(new_tensor)
#         predicted_time = y_scaler.inverse_transform(prediction.numpy()).flatten()[0]

#     print( str(DRIVER_NUMBERS[i]) + " Predicted average lap time:", predicted_time)

# Loss plot
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.show()






# Test RMSE: 3.2
# Test MAE: 2.6
# Accuracy within ±2 sec: 72%
# Accuracy within ±5 sec: 95%
# Then you can say "the model predicts average lap time with about 3-second RMSE error,"
# "and 95% of predictions are within 5 seconds of the actual value",
# which is very good, especially for real-world sports data where variance is natural.


#My Updates:
                        #Current
        # Epoch 100: Train Loss = 0.0085, Val Loss = 0.0022

        # Test MSE: 61.7526
        # Test RMSE: 7.8583
        # Test MAE: 4.9267
        # Accuracy within ±1 sec: 15.99%
        # Accuracy within ±2 sec: 30.41%
        # Accuracy within ±5 sec: 68.34%
        # Accuracy within ±10 sec: 89.97%


        # StandardScaler (z-score) for the target, and apply .inverse_transform() after prediction
    
        # Before: batch_size = 32
        # Now: batch_size = 16

        # Before: lr = 0.001
        # Now: lr = 0.0005

        # Dropout of 0.1 added to avoid overfitting
        # Final architecture: Linear → ReLU → Dropout → Linear → ReLU → Linear

        # Predicts lap times with ~3s MAE
        # Hits ~85% accuracy within ±5s
        # Uses statistically sound preprocessing and postprocessing

        # Epoch 100: Train Loss = 0.0171, Val Loss = 0.0094

        # Test MSE: 28.5087
        # Test RMSE: 5.3393
        # Test MAE: 2.9003
        # Accuracy within ±1s: 27.59%
        # Accuracy within ±2s: 52.35%
        # Accuracy within ±5s: 86.52%
        # Accuracy within ±10s: 96.87%


        #criterion = nn.SmoothL1Loss() - Slight improvement over MSE (Doesnt affect MSE at the end)
        # Epoch 100: Train Loss = 0.0076, Val Loss = 0.0047

        # Test MSE: 27.4075
        # Test RMSE: 5.2352
        # Test MAE: 2.6957
        # Accuracy within ±1s: 29.78%
        # Accuracy within ±2s: 59.87%
        # Accuracy within ±5s: 88.71%
        # Accuracy within ±10s: 97.81%


#Cant make Predictions on making_predictions.csv as doesnt includes driver names

