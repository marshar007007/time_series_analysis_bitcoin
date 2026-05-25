# Bitcoin Price Forecasting — Multivariate Time Series Analysis

A comparison of classical and deep learning approaches for forecasting Bitcoin's daily closing price using 8 market features over the period 2021–2026.

**Team:** San San Maw · Peter Ling Maung · Thandar Htwe

---

## Overview

This project explores whether modern deep learning models can outperform a classical statistical baseline for one-step-ahead Bitcoin price prediction. We train and evaluate four models on the same dataset and test set:

| Model | Type |
|---|---|
| ARIMA(1,1,0) | Classical univariate baseline |
| Multivariate LSTM | Deep learning — forward sequence |
| Multivariate BiLSTM | Deep learning — bidirectional sequence |
| Hybrid ARIMA + MV LSTM | Combined linear + non-linear |

The key finding: **ARIMA is a surprisingly strong baseline** (R² ≈ 0.97, MAPE ≈ 1.3%), and the **Hybrid model achieves the best overall performance** by combining ARIMA's linear forecast with an LSTM trained on the residuals.

---

## Dataset

**Bitcoin Market Analysis Dataset (2021–2025)**
- 1,826 daily observations from January 2021 to January 2026
- Target variable: `close` — daily closing price in USD
- 7 external features used as LSTM inputs:

| Feature | Description |
|---|---|
| `volume` | Daily trading volume |
| `sp500_close` | S&P 500 closing price |
| `gold_close` | Gold closing price |
| `dxy_close` | US Dollar Index |
| `fng_score` | Fear & Greed Index (0–100) |
| `hash_rate` | Bitcoin network hash rate |
| `google_trends_score` | Google search interest score |

> Place the CSV file in the project root before running the notebook. The expected filename is `Bitcoin Market Analysis Dataset (2021-2025).csv`.

---

## Project Structure

```
├── bitcoin_multivariate_time_series_analysis.ipynb   # Main notebook
├── Bitcoin Market Analysis Dataset (2021-2025).csv   # Dataset (add manually)
├── 01_raw_price_and_returns.png
├── 01b_all_features.png
├── 02_transformations.png
├── 03_acf_pacf.png
├── 04_train_test_split.png
├── 05_aic_bic_comparison.png
├── 06_arima_insample_fit.png
├── 07_residual_diagnostics.png
├── 08_qq_plot.png
├── 09_arima_forecast.png
├── 10_correlation_heatmap.png
├── 11_mv_lstm_loss.png
├── 12_mv_bilstm_loss.png
├── 13_all_forecasts.png
├── 14_metrics_comparison.png
└── 15_predicted_vs_actual.png
```

---

## Methodology

### 1. Stationarity Testing
The raw price series fails the ADF test (non-stationary). Log returns — the first difference of log price — pass at p < 0.05 and are used as the ARIMA input, confirming d = 1.

### 2. ARIMA Model Selection
ACF and PACF plots of the stationary series, combined with a grid search over p, d, q ∈ {0, 1, 2} ranked by AIC and BIC, both select **ARIMA(1, 1, 0)** as the best model. Residual diagnostics (Ljung-Box, Q-Q plot) confirm the residuals are white noise.

### 3. Train / Test Split
The data is split temporally — no shuffling — using 85% for training (Jan 2021 – Apr 2025) and 15% for testing (Apr 2025 – Jan 2026). The ARIMA forecast uses a **rolling one-step-ahead** strategy, refitting at each step with the true price history.

### 4. LSTM & BiLSTM
Both models use a 14-day sliding window across all 8 features as input (shape: `samples × 14 × 8`). The MinMaxScaler is fitted on training data only to prevent leakage. Architecture: two stacked LSTM/BiLSTM layers (100 + 50 units) followed by Dense layers.

### 5. Hybrid Model
The hybrid model works in two stages:
1. ARIMA produces a rolling forecast.
2. An LSTM is trained on the **ARIMA residuals** (actual − forecast) using the 7 external features as context.

The final prediction is `ARIMA forecast + LSTM residual correction`. Critically, test residuals are derived from the ARIMA forecast only — not actual prices — so no ground-truth information leaks into the LSTM.

---

## Results

Evaluated on the held-out test set (Apr 2025 – Jan 2026):

| Model | RMSE (USD) | MAE (USD) | R² |
|---|---|---|---|
| ARIMA(1,1,0) | ~1,500 | ~960 | ~0.970 |
| Multivariate LSTM | ~2,800 | ~2,100 | ~0.930 |
| Multivariate BiLSTM | ~2,750 | ~2,050 | ~0.932 |
| **Hybrid ARIMA + MV LSTM** | **~1,450** | **~920** | **~0.973** |

The Hybrid model achieves the best score across all metrics. The standalone LSTM and BiLSTM underperform the ARIMA baseline, reflecting both the near-random-walk nature of daily Bitcoin returns and the train/validation loss gap observed during training.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
statsmodels
scipy
scikit-learn
tensorflow >= 2.x
ipython
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn statsmodels scipy scikit-learn tensorflow ipython
```

---

## How to Run

1. Clone the repo and add the dataset CSV to the project root.
2. Install dependencies (see above).
3. Open `bitcoin_multivariate_time_series_analysis.ipynb` in Jupyter or VS Code.
4. Run all cells from top to bottom. The notebook will save all plots to the current directory.

> **Note:** The ARIMA rolling forecast loop refits the model at each test step and may take a few minutes to complete depending on your hardware.

---

## Limitations

- All models produce **one-step-ahead forecasts only**. Multi-day horizons are not covered.
- The 14-day lookback window was not systematically tuned.
- The LSTM models show a train/validation loss gap, suggesting some overfitting that early stopping alone did not fully resolve.
- An outlier in early 2021 affects the normality of ARIMA residuals.

---

## License

This project is for academic purposes. Dataset usage is subject to its original source terms.
