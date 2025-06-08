import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import haversine_distances
import holidays
import argparse
from tqdm import tqdm
import os
import random

# Set global seed for reproducibility
SEED = 40

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# For CUDA, if you use it
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# PyTorch deterministic settings (for complete reproducibility, can slow down training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Optional: set PYTHONHASHSEED for Python's hash-based operations (e.g., shuffling dict keys)
os.environ["PYTHONHASHSEED"] = str(SEED)

# --- Constants & Config ---
HISTORY_LEN = 48
PRED_HORIZON = 4
K_NEIGHBORS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 8
HIDDEN_DIM = 64
N_LAYERS = 4
LR = 0.0005
WEIGHT_DECAY = 0.0001
DROPOUT = 0.2
EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 128
VAL_FRAC = 0.2
STRIDE = 1
MODEL_PATH = "tcn_model_final_with_stats.pt"
WEATHER_FEATURES = ['temperature_2m', 'precipitation', 'windspeed_10m', 'cloudcover']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Dataset ---
class SharedTCNDataset(Dataset):
    def __init__(self, df, station_cols, neighbors, history_len, pred_horizon, weather_features, sample_indices):
        self.samples = []
        self.station_to_idx = {name: i for i, name in enumerate(station_cols)}
        timestamps = pd.to_datetime(df['timestamp'])
        hour_sin = np.sin(2 * np.pi * timestamps.dt.hour / 24)
        hour_cos = np.cos(2 * np.pi * timestamps.dt.hour / 24)
        dow_sin = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
        dow_cos = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
        month_sin = np.sin(2 * np.pi * timestamps.dt.month / 12)
        month_cos = np.cos(2 * np.pi * timestamps.dt.month / 12)
        is_weekend = (timestamps.dt.dayofweek >= 5).astype(float)
        slo_holidays = holidays.Slovenia()
        is_holiday = timestamps.dt.date.astype(str).isin([str(d) for d in slo_holidays]).astype(float)
        weather_array = df[weather_features].values
        time_feats = np.concatenate([
            np.stack([hour_sin, hour_cos, dow_sin, dow_cos,
                      month_sin, month_cos, is_weekend, is_holiday], axis=1),
            weather_array
        ], axis=1)
        bikes = df[station_cols].values.astype(np.float32)
        N = len(df)
        for s_name in station_cols:
            s_idx = self.station_to_idx[s_name]
            nn_idx = [self.station_to_idx[nn] for nn in neighbors[s_name]]
            series = bikes[:, [s_idx] + nn_idx]
            full_feats = np.concatenate([series, time_feats], axis=1)
            for i in sample_indices:
                x = full_feats[i - history_len:i]
                y = bikes[i:i + pred_horizon, s_idx]
                self.samples.append((x, y, s_idx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y, sid = self.samples[idx]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(sid, dtype=torch.long))

# --- Model ---
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, num_stations, embed_dim):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, dilation_size, dropout)]
        self.tcn = nn.Sequential(*layers)
        self.embedding = nn.Embedding(num_stations, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1] + embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    def forward(self, x, station_id):
        x = x.permute(0, 2, 1)
        tcn_out = self.tcn(x)[:, :, -1]
        emb = self.embedding(station_id)
        combined = torch.cat([tcn_out, emb], dim=1)
        return self.head(combined)

# --- Utility Functions ---
def load_and_clean_bikes(filename):
    df = pd.read_csv(filename)
    station_cols = df.columns[1:]
    for col in station_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[station_cols] = df[station_cols].ffill().bfill()
    df = df.dropna(subset=station_cols, how='all').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    return df, station_cols

def load_weather(filename):
    weather_df = pd.read_csv(filename, skiprows=2)
    weather_df = weather_df.rename(columns={
        'temperature_2m (°C)': 'temperature_2m',
        'precipitation (mm)': 'precipitation',
        'windspeed_10m (km/h)': 'windspeed_10m',
        'cloudcover (%)': 'cloudcover'
    })
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    return weather_df

def merge_weather(df, weather_df):
    return pd.merge(df, weather_df, left_on='timestamp', right_on='time', how='left')

def calc_neighbors(meta_csv, k=K_NEIGHBORS):
    meta = pd.read_csv(meta_csv)
    coords = np.deg2rad(meta[['latitude', 'longitude']].values)
    station_names = meta['name'].tolist()
    dists = haversine_distances(coords, coords) * 6371
    neighbors = {}
    for i, name in enumerate(station_names):
        order = np.argsort(dists[i])
        nn_idx = [j for j in order if j != i][:k]
        neighbors[name] = [station_names[j] for j in nn_idx]
    return neighbors

def make_sample_indices(mask, history_len, pred_horizon, stride=1):
    N = len(mask)
    indices = []
    for i in range(history_len, N - pred_horizon + 1, stride):
        if mask[i - history_len:i + pred_horizon].all():
            indices.append(i)
    return indices

def compute_time_features(timestamps, slo_holidays=None):
    hour_sin = np.sin(2 * np.pi * timestamps.dt.hour / 24)
    hour_cos = np.cos(2 * np.pi * timestamps.dt.hour / 24)
    dow_sin = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
    month_sin = np.sin(2 * np.pi * timestamps.dt.month / 12)
    month_cos = np.cos(2 * np.pi * timestamps.dt.month / 12)
    is_weekend = (timestamps.dt.dayofweek >= 5).astype(float)
    if slo_holidays is None:
        slo_holidays = holidays.Slovenia()
    is_holiday = timestamps.dt.date.astype(str).isin([str(d) for d in slo_holidays]).astype(float)
    time_feats = np.stack([hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_weekend, is_holiday], axis=1)
    return time_feats

# --- Train ---
def train_main():
    df, station_cols = load_and_clean_bikes("data/bicikelj_train.csv")
    weather_df = load_weather("data/weather_ljubljana.csv")
    df_merged = merge_weather(df, weather_df)
    df_merged[WEATHER_FEATURES] = df_merged[WEATHER_FEATURES].ffill().bfill()
    N = len(df_merged)

    # Validation split (non-overlapping blocks)
    BLOCK_SIZE = HISTORY_LEN + PRED_HORIZON
    #np.random.seed(42)
    all_possible_starts = np.arange(0, N - BLOCK_SIZE + 1)
    val_mask = np.zeros(N, dtype=bool)
    val_starts = []
    target_val_coverage = int(VAL_FRAC * N)
    covered = 0
    np.random.shuffle(all_possible_starts)
    for start in all_possible_starts:
        if val_mask[start:start + BLOCK_SIZE].any(): continue
        val_mask[start:start + BLOCK_SIZE] = True
        val_starts.append(start)
        covered += BLOCK_SIZE
        if covered >= target_val_coverage: break
    train_mask = ~val_mask

    train_indices = make_sample_indices(train_mask, HISTORY_LEN, PRED_HORIZON, stride=STRIDE)
    val_indices = make_sample_indices(val_mask, HISTORY_LEN, PRED_HORIZON, stride=1)

    # Normalization (train stats only)
    station_means = df_merged.loc[train_mask, station_cols].mean()
    station_stds  = df_merged.loc[train_mask, station_cols].std().replace(0, 1)
    weather_means = df_merged.loc[train_mask, WEATHER_FEATURES].mean()
    weather_stds  = df_merged.loc[train_mask, WEATHER_FEATURES].std().replace(0, 1)
    df_merged[station_cols] = (df_merged[station_cols] - station_means) / station_stds
    df_merged[WEATHER_FEATURES] = (df_merged[WEATHER_FEATURES] - weather_means) / weather_stds

    # Neighbors
    neighbors = calc_neighbors("data/bicikelj_metadata.csv", K_NEIGHBORS)

    # Datasets
    train_dataset = SharedTCNDataset(df_merged, station_cols, neighbors, HISTORY_LEN, PRED_HORIZON, WEATHER_FEATURES, train_indices)
    val_dataset = SharedTCNDataset(df_merged, station_cols, neighbors, HISTORY_LEN, PRED_HORIZON, WEATHER_FEATURES, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model
    input_size = 1 + K_NEIGHBORS + (8 + len(WEATHER_FEATURES))
    model = TCN(input_size, PRED_HORIZON, [HIDDEN_DIM] * N_LAYERS, 3, DROPOUT,
                num_stations=len(station_cols), embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb, sid in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            xb, yb, sid = xb.to(DEVICE), yb.to(DEVICE), sid.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb, sid), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, sid in val_loader:
                xb, yb, sid = xb.to(DEVICE), yb.to(DEVICE), sid.to(DEVICE)
                val_loss += criterion(model(xb, sid), yb).item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping!")
                break

    model.load_state_dict(best_state)
    torch.save({
        'model': model.state_dict(),
        'station_means': station_means,
        'station_stds': station_stds,
        'weather_means': weather_means,
        'weather_stds': weather_stds
    }, MODEL_PATH)
    print(f"✅ Saved model to '{MODEL_PATH}'")

# --- Predict ---
def predict_main():
    print("🚲 Prediction mode")
    # --- Load test and train stats for normalization ---
    df, station_cols = load_and_clean_bikes("data/bicikelj_train.csv")
    weather_train_df = load_weather("data/weather_ljubljana.csv")
    df_merged = merge_weather(df, weather_train_df)
    df_merged[WEATHER_FEATURES] = df_merged[WEATHER_FEATURES].ffill().bfill()
    station_means = df_merged[station_cols].mean()
    station_stds = df_merged[station_cols].std().replace(0, 1)
    weather_means = df_merged[WEATHER_FEATURES].mean()
    weather_stds = df_merged[WEATHER_FEATURES].std().replace(0, 1)
    neighbors = calc_neighbors("data/bicikelj_metadata.csv", K_NEIGHBORS)
    name_to_idx = {name: i for i, name in enumerate(station_cols)}

    # Load model and normalization
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    input_size = 1 + K_NEIGHBORS + (8 + len(WEATHER_FEATURES))
    model = TCN(input_size, PRED_HORIZON, [HIDDEN_DIM] * N_LAYERS, 3, DROPOUT,
                num_stations=len(station_cols), embed_dim=EMBED_DIM).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # Use train stats saved with model, if present:
    station_means = checkpoint.get('station_means', station_means)
    station_stds = checkpoint.get('station_stds', station_stds)
    weather_means = checkpoint.get('weather_means', weather_means)
    weather_stds = checkpoint.get('weather_stds', weather_stds)

    # --- Prepare test set ---
    test_df = pd.read_csv("data/bicikelj_test.csv")
    station_cols = list(station_means.index)

    test_feats = test_df[station_cols].values.astype(np.float32)
    timestamps = pd.to_datetime(test_df["timestamp"])
    weather_test_df = load_weather("data/weather_ljubljana_test.csv")
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp']).dt.tz_localize(None)
    test_df_merged = merge_weather(test_df, weather_test_df)
    test_df_merged[WEATHER_FEATURES] = test_df_merged[WEATHER_FEATURES].ffill().bfill()
    weather_feats = test_df_merged[WEATHER_FEATURES].values.astype(np.float32)
    time_feats = compute_time_features(timestamps)
    test_feats_norm = (test_feats - station_means.values) / station_stds.values
    weather_feats_norm = (weather_feats - weather_means.values) / weather_stds.values

    pred_matrix = np.full_like(test_feats, np.nan)
    with torch.no_grad():
        for i in range(HISTORY_LEN, len(test_df) - PRED_HORIZON + 1):
            # Only predict for rows that are missing all stations (== test target region)
            if np.isnan(test_feats[i:i + PRED_HORIZON]).all(axis=0).all():
                for station in station_cols:
                    s_idx = name_to_idx[station]
                    nn_idx = [name_to_idx[nn] for nn in neighbors[station]]
                    seq = []
                    for t in range(i - HISTORY_LEN, i):
                        row = [test_feats_norm[t, s_idx]]
                        row += [test_feats_norm[t, j] for j in nn_idx]
                        row += list(time_feats[t])
                        row += list(weather_feats_norm[t])
                        seq.append(row)
                    seq = torch.tensor([seq], dtype=torch.float32).to(DEVICE)
                    sid_tensor = torch.tensor([s_idx], dtype=torch.long, device=DEVICE)
                    pred_norm = model(seq, sid_tensor).cpu().numpy().flatten()
                    # When using Series, always access by station name:
                    pred = pred_norm * station_stds[station] + station_means[station]
                    for j in range(PRED_HORIZON):
                        pred_matrix[i + j, s_idx] = pred[j]
    # --- Save predictions ---
    pred_df = pd.DataFrame(pred_matrix, columns=station_cols)
    pred_df.insert(0, "timestamp", test_df["timestamp"])
    rows_to_output = test_df[station_cols].isna().all(axis=1)
    pred_df_filtered = pred_df[rows_to_output].copy()
    pred_df_filtered.to_csv("bicikelj_test_predictions_tcn_weather.csv", index=False)
    print("✅ Saved predictions to 'bicikelj_test_predictions_tcn_weather.csv'")

# --- Main Entrypoint ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train mode: trains and saves model')
    args = parser.parse_args()
    if args.train:
        train_main()
    else:
        predict_main()

if __name__ == "__main__":
    main()
