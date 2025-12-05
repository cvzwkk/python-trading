
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ML Libraries
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models

# Kalman
from pykalman import KalmanFilter


# ==========================================================
# Fetch OHLCV Bars
# ==========================================================
def fetch_ohlcv_bars(exchange_id="binanceus", symbol="BTC/USDT", timeframe="1m", limit=60):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    return df


# ==========================================================
# VWAP + Standard Deviation Bands
# ==========================================================
def compute_vwap_std(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].fillna(0)

    vwap = (tp * vol).sum() / vol.sum()
    std1 = np.sqrt(((tp - vwap) ** 2).mean())
    std2 = std1 * 2

    return {
        "VWAP": vwap,
        "Upper 1σ": vwap + std1,
        "Upper 2σ": vwap + std2,
        "Lower 1σ": vwap - std1,
        "Lower 2σ": vwap - std2,
        "Latest datetime": df["datetime"].iloc[-1],
        "Latest close": df["close"].iloc[-1]
    }


# ==========================================================
# Feature Engineering
# ==========================================================
def prepare_features(df):
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["volatility"] = df["return"].rolling(20).std()
    df["target"] = df["close"].shift(-1)

    df = df.dropna()

    features = ["close", "ma20", "ma50", "volatility", "volume"]
    X = df[features]
    y = df["target"]

    return df, X, y


# ==========================================================
# ML Models
# ==========================================================
def model_ridge(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(Xs, y, test_size=0.2)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict([Xs[-1]])[0]


def model_xgboost(X, y):
    model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5)
    model.fit(X, y)
    return model.predict([X.iloc[-1]])[0]


def model_gbm(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model.predict([X.iloc[-1]])[0]


# ==========================================================
# Kalman Filter (Smoothed last value)
# ==========================================================
def apply_kalman(df):
    kf = KalmanFilter(
        initial_state_mean=df["close"].iloc[0],
        n_dim_obs=1,
        n_dim_state=1
    )
    smoothed, _ = kf.smooth(df["close"].values)
    return float(smoothed[-1])


# ==========================================================
# Autoencoder (Latent Factors)
# ==========================================================
def model_autoencoder(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    input_dim = Xs.shape[1]

    encoder = models.Sequential([
        layers.Dense(4, activation='relu'),
        layers.Dense(2, activation='linear', name="latent")
    ])

    decoder = models.Sequential([
        layers.Dense(4, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])

    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(Xs, Xs, epochs=20, verbose=0)

    latent_vec = encoder.predict(Xs[-1:])
    return latent_vec[0]


# ==========================================================
# GRU Model
# ==========================================================
def model_gru(df):
    seq = 30
    data = df["close"].values

    X, Y = [], []
    for i in range(len(data) - seq):
        X.append(data[i:i+seq])
        Y.append(data[i+seq])

    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)

    model = models.Sequential([
        layers.GRU(32),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)

    return model.predict(X[-1].reshape(1, seq, 1))[0][0]


# ==========================================================
# Transformer
# ==========================================================
def model_transformer(df):
    seq = 30
    data = df["close"].values

    X, Y = [], []
    for i in range(len(data) - seq):
        X.append(data[i:i + seq])
        Y.append(data[i + seq])

    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)

    inp = layers.Input(shape=(seq, 1))
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)(inp, inp)
    x = layers.LayerNormalization()(inp + attn)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)

    return model.predict(X[-1].reshape(1, seq, 1))[0][0]

# ==========================================================
# CNN + LSTM Hybrid Model
# ==========================================================
def model_cnn_lstm(df):
    seq = 30
    data = df["close"].values

    X, Y = [], []
    for i in range(len(data) - seq):
        X.append(data[i:i + seq])
        Y.append(data[i + seq])

    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)

    model = models.Sequential([
        # 1D CNN extracts local candle features
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding="causal", input_shape=(seq, 1)),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding="causal"),
        layers.MaxPooling1D(pool_size=2),

        # LSTM captures long-term temporal patterns
        layers.LSTM(32),

        # Final linear forecast
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)

    return model.predict(X[-1].reshape(1, seq, 1))[0][0]

# ==========================================================
# Bidirectional LSTM
# ==========================================================
def model_bilstm(df):
    seq = 30
    data = df["close"].values

    X, Y = [], []
    for i in range(len(data) - seq):
        X.append(data[i:i+seq])
        Y.append(data[i+seq])

    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)

    model = models.Sequential([
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)

    return float(model.predict(X[-1].reshape(1, seq, 1))[0][0])

# ==========================================================
# LSTM + Attention Hybrid
# ==========================================================
def model_lstm_attention(df):
    seq = 30
    data = df["close"].values

    X, Y = [], []
    for i in range(len(data) - seq):
        X.append(data[i:i+seq])
        Y.append(data[i+seq])

    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)

    inp = layers.Input(shape=(seq, 1))
    lstm_out = layers.LSTM(32, return_sequences=True)(inp)
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)(lstm_out, lstm_out)
    x = layers.GlobalAveragePooling1D()(attn)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)

    return float(model.predict(X[-1].reshape(1, seq, 1))[0][0])

# ==========================================================
# Temporal Convolution Network (TCN)
# ==========================================================
def model_tcn(df):
    seq = 30
    data = df["close"].values

    X, Y = [], []
    for i in range(len(data)-seq):
        X.append(data[i:i+seq])
        Y.append(data[i+seq])

    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)

    model = models.Sequential([
        layers.Conv1D(64, kernel_size=3, dilation_rate=1,
                      padding="causal", activation="relu", input_shape=(seq,1)),
        layers.Conv1D(64, kernel_size=3, dilation_rate=2,
                      padding="causal", activation="relu"),
        layers.Conv1D(64, kernel_size=3, dilation_rate=4,
                      padding="causal", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)

    return float(model.predict(X[-1].reshape(1,seq,1))[0][0])

# ==========================================================
# CNN + GRU Hybrid
# ==========================================================
def model_cnn_gru(df):
    seq = 30
    data = df["close"].values

    X, Y = [], []
    for i in range(len(data)-seq):
        X.append(data[i:i+seq])
        Y.append(data[i+seq])

    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)

    model = models.Sequential([
        layers.Conv1D(32, 3, activation='relu', padding="causal", input_shape=(seq,1)),
        layers.Conv1D(32, 3, activation='relu', padding="causal"),
        layers.MaxPooling1D(2),
        layers.GRU(32),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)

    return float(model.predict(X[-1].reshape(1,seq,1))[0][0])


# ==========================================================
# Deep MLP (Features)
# ==========================================================
def model_deep_mlp(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = models.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(Xs, y, epochs=20, verbose=0)

    return float(model.predict(Xs[-1].reshape(1, -1))[0][0])

# ==========================================================
# Seq2Seq Encoder-Decoder LSTM
# ==========================================================
def model_seq2seq(df):
    seq = 30
    data = df["close"].values

    X, Y = [], []
    for i in range(len(data)-seq):
        X.append(data[i:i+seq])
        Y.append(data[i+seq])

    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)

    inp = layers.Input(shape=(seq,1))
    enc = layers.LSTM(32, return_state=True)
    enc_out, h, c = enc(inp)

    dec_inp = layers.RepeatVector(1)(enc_out)
    dec_out = layers.LSTM(32)(dec_inp)

    out = layers.Dense(1)(dec_out)

    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)

    return float(model.predict(X[-1].reshape(1,seq,1))[0][0])

# ==========================================================
# Random Forest Regressor
# ==========================================================
from sklearn.ensemble import RandomForestRegressor

def model_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=300)
    model.fit(X, y)
    return float(model.predict([X.iloc[-1]])[0])




# ==========================================================
# Trend Signal Function
# ==========================================================
def trend_signal(pred, last_price):
    if pred > last_price:
        return "BULLISH 📈"
    elif pred < last_price:
        return "BEARISH 📉"
    return "NEUTRAL ➖"


# ==========================================================
# MAIN EXECUTION
# ==========================================================
symbol = "BTC/USDT"
exchange_id = "binanceus"

bars = fetch_ohlcv_bars(exchange_id, symbol, timeframe="1w", limit=224)
summary = compute_vwap_std(bars)
df, X, y = prepare_features(bars)

# Predictions
ridge_pred = model_ridge(X, y)
xgb_pred = model_xgboost(X, y)
gbm_pred = model_gbm(X, y)
kalman_pred = apply_kalman(df)
ae_latent = model_autoencoder(X)
gru_pred = model_gru(df)
transformer_pred = model_transformer(df)
cnn_lstm_pred = model_cnn_lstm(df)
cnn_lstm_signal = trend_signal(cnn_lstm_pred, last_close)
bilstm_pred = model_bilstm(df)
attention_pred = model_lstm_attention(df)
tcn_pred = model_tcn(df)
cnn_gru_pred = model_cnn_gru(df)
mlp_pred = model_deep_mlp(X, y)
seq2seq_pred = model_seq2seq(df)
rf_pred = model_random_forest(X, y)

last_close = df["close"].iloc[-1]

# Signals
ridge_signal = trend_signal(ridge_pred, last_close)
xgb_signal = trend_signal(xgb_pred, last_close)
gbm_signal = trend_signal(gbm_pred, last_close)
kalman_signal = trend_signal(kalman_pred, last_close)
gru_signal = trend_signal(gru_pred, last_close)
transformer_signal = trend_signal(transformer_pred, last_close)
bilstm_signal = trend_signal(bilstm_pred, last_close)
attention_signal = trend_signal(attention_pred, last_close)
tcn_signal = trend_signal(tcn_pred, last_close)
cnn_gru_signal = trend_signal(cnn_gru_pred, last_close)
mlp_signal = trend_signal(mlp_pred, last_close)
seq2seq_signal = trend_signal(seq2seq_pred, last_close)
rf_signal = trend_signal(rf_pred, last_close)


# ==========================================================
# PRINT OUTPUT
# ==========================================================
print("\n=== VWAP Summary ===")
for k, v in summary.items():
    print(f"{k:15}: {v}")

print("\n=== Machine Learning Forecasts ===")
print(f"Current Close:        {last_close:.2f}\n")

print(f"Ridge Regression:     {ridge_pred:.2f}   → {ridge_signal}")
print(f"XGBoost:              {xgb_pred:.2f}   → {xgb_signal}")
print(f"GBM:                  {gbm_pred:.2f}   → {gbm_signal}")
print(f"Kalman Smoothed:      {kalman_pred:.2f}   → {kalman_signal}")
print(f"GRU Forecast:         {gru_pred:.2f}   → {gru_signal}")
print(f"Transformer Forecast: {transformer_pred:.2f}   → {transformer_signal}")
print(f"CNN-LSTM Forecast:    {cnn_lstm_pred:.2f}   → {cnn_lstm_signal}")
print(f"BiLSTM Forecast:       {bilstm_pred:.2f} → {bilstm_signal}")
print(f"LSTM-Attention:        {attention_pred:.2f} → {attention_signal}")
print(f"TCN Forecast:          {tcn_pred:.2f} → {tcn_signal}")
print(f"CNN-GRU Forecast:      {cnn_gru_pred:.2f} → {cnn_gru_signal}")
print(f"Deep MLP Forecast:     {mlp_pred:.2f} → {mlp_signal}")
print(f"Seq2Seq Forecast:      {seq2seq_pred:.2f} → {seq2seq_signal}")
print(f"Random Forest:         {rf_pred:.2f} → {rf_signal}")

print("\nAutoencoder latent factors:", ae_latent)
