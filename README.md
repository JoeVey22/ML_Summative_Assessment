# KIE4031 Machine Learning Final Summative Assessment

This folder contains an English-only solution for the stock price prediction assignment.

## Files

- `src/train_stock_rnn.py` - source code for data download, preprocessing, LSTM training, baseline comparison, metrics, and visualizations.
- `data/googl_daily_2019_2025.csv` - downloaded public Alphabet Class A (GOOGL) stock dataset from Yahoo Finance chart data.
- `outputs/metrics.json` - final evaluation metrics.
- `outputs/model_comparison.csv` - comparison of RNN, GRU, LSTM, and moving-average baseline.
- `outputs/test_predictions.csv` - actual vs predicted test values.
- `outputs/stock_price_history.png` - full adjusted close history with train/validation/test split markers.
- `outputs/prediction_vs_actual.png` - test set prediction graph.
- `outputs/training_loss.png` - training loss curves.
- `outputs/validation_loss.png` - validation loss curves.
- `outputs/loss_history.csv` - epoch-by-epoch training and validation losses.
- `outputs/model_rmse_comparison.png` - RMSE comparison chart.
- `report.md` - assignment report in English.

## How to Run

```powershell
python src\train_stock_rnn.py
```

The script will reuse `data/googl_daily_2019_2025.csv` if it already exists. Delete the CSV if you want to download fresh Yahoo Finance data.

## Source Code Link

If a URL is required for submission, upload this folder to GitHub or Google Drive and use that share link. The main source file is `src/train_stock_rnn.py`.
