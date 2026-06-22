import csv
import json
import math
import random
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_PATH = DATA_DIR / "googl_daily_2019_2025.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

SYMBOL = "GOOGL"
START_DATE = "2019-01-01"
END_DATE = "2025-12-31"
LOOKBACK_DAYS = 60
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.20
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
SEED = 42


class StockRecurrentModel(nn.Module):
    def __init__(self, cell_type, input_size=1, hidden_size=64, num_layers=2, dropout=0.20):
        super().__init__()
        recurrent_layers = {
            "RNN": nn.RNN,
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
        }
        self.recurrent = recurrent_layers[cell_type](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        output, _ = self.recurrent(x)
        return self.regressor(output[:, -1, :])


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unix_timestamp(date_text):
    dt = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def download_yahoo_chart(symbol, start_date, end_date):
    period1 = unix_timestamp(start_date)
    period2 = unix_timestamp(end_date)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d"
        "&events=history&includeAdjustedClose=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adjclose = result["indicators"]["adjclose"][0]["adjclose"]

    rows = []
    for i, timestamp in enumerate(timestamps):
        if adjclose[i] is None:
            continue
        rows.append(
            {
                "Date": datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d"),
                "Open": quote["open"][i],
                "High": quote["high"][i],
                "Low": quote["low"][i],
                "Close": quote["close"][i],
                "Adj Close": adjclose[i],
                "Volume": quote["volume"][i],
            }
        )

    DATA_DIR.mkdir(exist_ok=True)
    with DATA_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return DATA_PATH


def load_and_clean_data():
    if not DATA_PATH.exists():
        download_yahoo_chart(SYMBOL, START_DATE, END_DATE)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")
    df = df.dropna(subset=["Adj Close"])
    df = df[df["Adj Close"] > 0].reset_index(drop=True)
    return df


def build_sequences(values, lookback):
    x, y = [], []
    for idx in range(lookback, len(values)):
        x.append(values[idx - lookback : idx])
        y.append(values[idx])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def inverse_scale(scaler, values):
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)


def evaluate(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
    }


def train_recurrent_model(model_name, x_train, y_train, x_val, y_val, x_test, scaler, device):
    set_reproducibility(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    val_x = torch.tensor(x_val).to(device)
    val_y = torch.tensor(y_val).to(device)

    model = StockRecurrentModel(model_name).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x)
            val_loss = loss_fn(val_prediction, val_y)
            val_losses.append(float(val_loss.item()))
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_epoch = epoch + 1
                best_state = deepcopy(model.state_dict())

        print(
            f"{model_name} Epoch {epoch + 1:02d}/{EPOCHS} - "
            f"train loss: {train_losses[-1]:.6f} - val loss: {val_losses[-1]:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        predicted_scaled = model(torch.tensor(x_test).to(device)).cpu().numpy().reshape(-1)

    predictions = inverse_scale(scaler, predicted_scaled)
    return predictions, train_losses, val_losses, best_epoch, best_val_loss


def main():
    set_reproducibility(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_and_clean_data()
    train_end_index = int(len(df) * (1 - VALIDATION_RATIO - TEST_RATIO))
    validation_end_index = int(len(df) * (1 - TEST_RATIO))

    scaler = MinMaxScaler()
    train_close = df.loc[: train_end_index - 1, ["Adj Close"]]
    scaler.fit(train_close)
    scaled_close = scaler.transform(df[["Adj Close"]])

    x_all, y_all = build_sequences(scaled_close, LOOKBACK_DAYS)
    sequence_dates = df["Date"].iloc[LOOKBACK_DAYS:].reset_index(drop=True)

    train_count = train_end_index - LOOKBACK_DAYS
    validation_count = validation_end_index - train_end_index
    x_train, y_train = x_all[:train_count], y_all[:train_count]
    x_val = x_all[train_count : train_count + validation_count]
    y_val = y_all[train_count : train_count + validation_count]
    x_test, y_test = x_all[train_count + validation_count :], y_all[train_count + validation_count :]
    test_dates = sequence_dates.iloc[train_count + validation_count :].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_test_actual = inverse_scale(scaler, y_test.reshape(-1))

    model_predictions = {}
    model_losses = {}
    model_val_losses = {}
    model_metrics = {}
    model_training_summary = {}
    for model_name in ["RNN", "GRU", "LSTM"]:
        predictions, losses, val_losses, best_epoch, best_val_loss = train_recurrent_model(
            model_name, x_train, y_train, x_val, y_val, x_test, scaler, device
        )
        model_predictions[model_name] = predictions
        model_losses[model_name] = losses
        model_val_losses[model_name] = val_losses
        model_metrics[model_name.lower()] = evaluate(y_test_actual, predictions)
        model_training_summary[model_name.lower()] = {
            "best_validation_epoch": int(best_epoch),
            "best_validation_loss": float(best_val_loss),
        }

    close_values = df["Adj Close"].to_numpy()
    naive_predictions = []
    for date in test_dates:
        date_index = df.index[df["Date"] == date][0]
        naive_predictions.append(np.mean(close_values[date_index - LOOKBACK_DAYS : date_index]))
    naive_predictions = np.array(naive_predictions)

    baseline_metrics = evaluate(y_test_actual, naive_predictions)

    metrics = {
        "symbol": SYMBOL,
        "dataset_source": "Yahoo Finance chart API",
        "date_range": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        "rows_after_cleaning": int(len(df)),
        "lookback_days": LOOKBACK_DAYS,
        "train_rows": int(train_count),
        "validation_rows": int(len(y_val)),
        "test_rows": int(len(y_test_actual)),
        "device": str(device),
        "models": model_metrics,
        "training_summary": model_training_summary,
        "moving_average_baseline": baseline_metrics,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    results = pd.DataFrame(
        {
            "Date": test_dates,
            "Actual": y_test_actual,
            "RNN_Predicted": model_predictions["RNN"],
            "GRU_Predicted": model_predictions["GRU"],
            "LSTM_Predicted": model_predictions["LSTM"],
            "Moving_Average_Baseline": naive_predictions,
        }
    )
    results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    loss_history = pd.DataFrame({"Epoch": range(1, EPOCHS + 1)})
    for model_name in ["RNN", "GRU", "LSTM"]:
        loss_history[f"{model_name}_Train_Loss"] = model_losses[model_name]
        loss_history[f"{model_name}_Validation_Loss"] = model_val_losses[model_name]
    loss_history.to_csv(OUTPUT_DIR / "loss_history.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Adj Close"], label="Adjusted close", color="#1f77b4")
    plt.axvline(df.loc[train_end_index, "Date"], color="#2ca02c", linestyle="--", label="Train/validation split")
    plt.axvline(df.loc[validation_end_index, "Date"], color="#d62728", linestyle="--", label="Validation/test split")
    plt.title(f"{SYMBOL} adjusted close price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stock_price_history.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, y_test_actual, label="Actual", color="#111111", linewidth=2)
    plt.plot(test_dates, model_predictions["RNN"], label="RNN prediction", color="#1f77b4", alpha=0.85)
    plt.plot(test_dates, model_predictions["GRU"], label="GRU prediction", color="#2ca02c", alpha=0.85)
    plt.plot(test_dates, model_predictions["LSTM"], label="LSTM prediction", color="#9467bd", alpha=0.85)
    plt.plot(test_dates, naive_predictions, label="60-day moving average baseline", color="#ff7f0e")
    plt.title(f"{SYMBOL} test set prediction")
    plt.xlabel("Date")
    plt.ylabel("Adjusted close price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_vs_actual.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    for model_name, losses in model_losses.items():
        plt.plot(range(1, EPOCHS + 1), losses, label=model_name)
    plt.title("RNN model training loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    for model_name, losses in model_val_losses.items():
        plt.plot(range(1, EPOCHS + 1), losses, label=model_name)
    plt.title("RNN model validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_loss.png", dpi=180)
    plt.close()

    comparison = pd.DataFrame(
        [
            {"Model": "RNN", **model_metrics["rnn"]},
            {"Model": "GRU", **model_metrics["gru"]},
            {"Model": "LSTM", **model_metrics["lstm"]},
            {"Model": "60-day Moving Average", **baseline_metrics},
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(comparison["Model"], comparison["RMSE"], color=["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"])
    plt.title("Model comparison by RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_rmse_comparison.png", dpi=180)
    plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
