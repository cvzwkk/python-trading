## Installing Modules  

```
!pip install ccxt  
!pip install mplfinance  
!pip install prophet==1.1.5 cmdstanpy==1.2.1  
 ```
 
## Listing avaible exchanges to use with ccxt  


```
import ccxt

exchanges = ccxt.exchanges
working = []

for ex_id in exchanges:
    try:
        ex = getattr(ccxt, ex_id)()
    ex.load_markets()
        working.append(ex_id)
    except Exception:
        pass

working[:40]  # show first 40 working free exchanges

```

## Printing VWAP and STDs based in desired timeframe and bars amount

```
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ---------------------------
# Fetch OHLCV bars
# ---------------------------
def fetch_ohlcv_bars(exchange_id="binanceus", symbol="BTC/USDT", timeframe="4h", limit=1000):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

# ---------------------------
# Compute VWAP and Std bands
# ---------------------------
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

# ---------------------------
# Main
# ---------------------------
symbol = "BTC/USDT"
exchange_id = "binanceus"

bars = fetch_ohlcv_bars(exchange_id, symbol, timeframe="1w", limit=224)
summary = compute_vwap_std(bars)

# ---------------------------
# Print table line by line
# ---------------------------
print(f"Summary for {symbol} based on last 50 × 4-hour bars:\n")
print(f"{'Metric':<15} | {'Value':>15}")
print("-"*33)
for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key:<15} | {value:>15.2f}")
    else:
        print(f"{key:<15} | {value}")

```

## Trend gattering   

```
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# Fetch OHLCV bars
# ---------------------------------------------------
def fetch_ohlcv_bars(exchange_id="binanceus", symbol="BTC/USDT", timeframe="1m", limit=200):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# ---------------------------------------------------
# VWAP + Std Bands + Regimes + Signals + Target
# ---------------------------------------------------
def compute_regime_and_signals(df):

    # --- VWAP core ---
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].fillna(0)

    vwap = (tp * vol).sum() / vol.sum()
    std1 = np.sqrt(((tp - vwap) ** 2).mean())
    std2 = std1 * 2

    latest_close = df["close"].iloc[-1]

    pvo = (latest_close - vwap) / std1  # Z-score deviation

    # --- Regimes ---
    if abs(latest_close - vwap) <= std1:
        regime = "Mean-Reversion"
        regime_factor = 0.5
    elif std1 < abs(latest_close - vwap) <= std2:
        regime = "Transition"
        regime_factor = 1.0
    else:
        regime = "Trend"
        regime_factor = 1.8

    # --- Trend Direction ---
    direction = "Bullish" if latest_close > vwap else "Bearish"

    # --- Forecast text ---
    if regime == "Mean-Reversion":
        forecast = "Reversion toward VWAP"
    elif regime == "Transition":
        forecast = "Prepare for breakout"
    else:
        forecast = f"Continuation ({direction})"

    # --- Confidence ---
    confidence = round(min(1, abs(pvo) / 3), 3)

    # ----------------------------------------------------------
    # SIGNALS
    # ----------------------------------------------------------
    signal = "HOLD"

    if regime == "Mean-Reversion":
        if latest_close <= vwap - std1:
            signal = "BUY"
        elif latest_close >= vwap + std1:
            signal = "SELL"

    elif regime == "Trend":
        if direction == "Bullish":
            signal = "BUY"
        elif direction == "Bearish":
            signal = "SELL"

    # ----------------------------------------------------------
    # STOP LOSS / TAKE PROFIT
    # ----------------------------------------------------------
    if signal == "BUY":
        stop_loss = vwap - std2
    elif signal == "SELL":
        stop_loss = vwap + std2
    else:
        stop_loss = None


    # ==========================================================
    # ADVANCED TARGET PRICE MODELS
    # ==========================================================

    # ----------------------------------------------------------
    # 1. ATR Volatility Target
    # ----------------------------------------------------------
    df["tr"] = np.maximum(df["high"] - df["low"],
                          np.maximum(abs(df["high"] - df["close"].shift(1)),
                                     abs(df["low"] - df["close"].shift(1))))

    atr = df["tr"].rolling(14).mean().iloc[-1]

    if signal == "BUY":
        atr_target = latest_close + atr * 1.2
    elif signal == "SELL":
        atr_target = latest_close - atr * 1.2
    else:
        atr_target = None

    # ----------------------------------------------------------
    # 2. VWAP statistical projection
    # ----------------------------------------------------------
    if signal == "BUY":
        vwap_target = vwap + std1 * regime_factor
    elif signal == "SELL":
        vwap_target = vwap - std1 * regime_factor
    else:
        vwap_target = None

    # ----------------------------------------------------------
    # 3. Linear Regression Forecast
    # ----------------------------------------------------------
    lr = LinearRegression()
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["close"].values
    lr.fit(X, y)

    next_index = np.array([[len(df) + 3]])   # 3 minutes ahead forecast
    lr_target = float(lr.predict(next_index))

    # ----------------------------------------------------------
    # Final blended target price
    # ----------------------------------------------------------
    if signal in ("BUY", "SELL"):
        target_final = (
            0.4 * atr_target +
            0.4 * vwap_target +
            0.2 * lr_target
        )
    else:
        target_final = None

    return {
        "VWAP": vwap,
        "Upper 1σ": vwap + std1,
        "Upper 2σ": vwap + std2,
        "Lower 1σ": vwap - std1,
        "Lower 2σ": vwap - std2,
        "Latest datetime": df["datetime"].iloc[-1],
        "Latest close": latest_close,
        "Deviation (PVO)": pvo,
        "Regime": regime,
        "Trend Direction": direction,
        "Forecast": forecast,
        "Confidence": confidence,
        "Signal": signal,
        "Stop-Loss": stop_loss,
        "ATR Target": atr_target,
        "VWAP Target": vwap_target,
        "LR Target": lr_target,
        "Final Target Price": target_final
    }


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
symbol = "BTC/USDT"
exchange_id = "binanceus"

bars = fetch_ohlcv_bars(exchange_id, symbol, timeframe="1w", limit=24)
summary = compute_regime_and_signals(bars)

# ---------------------------------------------------
# Print table
# ---------------------------------------------------
print(f"\nSummary for {symbol} (1m bars):\n")
print(f"{'Metric':<22} | Value")
print("-"*50)

for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key:<22} | {value:,.5f}")
    else:
        print(f"{key:<22} | {value}")


```
   
 ## Applying Trainer Models:       
 ** Training XGBoost/GBM...  **     
 ** stacking ensemble (sklearn) ... **      
 ** Autoencoder for feature compression...  **    
 ** GRU...   **   
 ** Transformer encoder.   **   
   
```
"""
Full upgraded ML pipeline: XGBoost, GRU, Transformer, Autoencoder, Kalman + stacking ensemble.
Safe imports: falls back gracefully if packages are missing.
Run locally. Keep GPU in mind for faster deep models.
"""

import os
import math
import warnings
warnings.filterwarnings("ignore")

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split

# ---------------------
# Safe optional imports
# ---------------------
HAS_XGBOOST = False
HAS_SKLEARN = True  # sklearn baseline required
HAS_TF = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    print("xgboost not available — will use sklearn GB if present (slower). Install: pip install xgboost")

try:
    from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score
except Exception:
    raise RuntimeError("scikit-learn is required. Install: pip install scikit-learn")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout, LayerNormalization, MultiHeadAttention, \
        GlobalAveragePooling1D, Conv1D, Flatten, Reshape
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception:
    HAS_TF = False
    print("TensorFlow not available — deep models (GRU/Transformer/Autoencoder) will be skipped. Install: pip install tensorflow")

# ---------------------
# Config
# ---------------------
EXCHANGE_ID = "binanceus"      # change if needed
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
LIMIT = 800                  # number of bars to fetch
HORIZON = 3                  # predict 3 steps ahead
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model training hyperparams (conservative / fast)
XGB_PARAMS = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
GB_PARAMS = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
GRU_PARAMS = {"units": 64, "epochs": 10, "batch": 32}
TRANSFORMER_PARAMS = {"d_model": 32, "num_heads": 2, "ff_dim": 64, "epochs": 8, "batch": 32}
AE_PARAMS = {"encoding_dim": 8, "epochs": 12, "batch": 32}

# ---------------------
# Utilities
# ---------------------
def fetch_ohlcv(exchange_id=EXCHANGE_ID, symbol=SYMBOL, timeframe=TIMEFRAME, limit=LIMIT):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    return df

def add_features(df):
    df = df.copy()
    # returns and log returns
    df["ret_1"] = df["close"].pct_change()
    df["logret_1"] = np.log(df["close"]).diff()
    # lag features
    for l in [1,2,3,5,10,20]:
        df[f"close_lag_{l}"] = df["close"].shift(l)
    # rolling stats
    df["rmean_5"] = df["close"].rolling(5).mean()
    df["rstd_5"] = df["close"].rolling(5).std()
    df["rmean_20"] = df["close"].rolling(20).mean()
    df["rstd_20"] = df["close"].rolling(20).std()
    # ATR (simple)
    prev = df["close"].shift(1)
    df["tr"] = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - prev), abs(df["low"] - prev)))
    df["atr_14"] = df["tr"].rolling(14).mean()
    # VWAP-like using typical price and volume for windowed VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_50"] = (tp * df["volume"]).rolling(50).sum() / (df["volume"].rolling(50).sum() + 1e-9)
    # momentum indicators
    df["mom_5"] = df["close"] - df["close"].shift(5)
    df["mom_10"] = df["close"] - df["close"].shift(10)
    # dropna
    df = df.dropna()
    return df

def build_regression_dataset(df, horizon=HORIZON):
    df2 = df.copy()
    df2[f"target_{horizon}"] = df2["close"].shift(-horizon)
    df2 = df2.dropna()
    X = df2.drop(columns=[c for c in df2.columns if c.startswith("target_") or c in ["timestamp", "open","high","low","close","volume"]])
    y = df2[f"target_{horizon}"].values
    return X, y, df2

# ---------------------
# Kalman filter simple trend estimator
# ---------------------
def simple_kalman_trend(prices, q=1e-5, r=0.001):
    # one-dimensional Kalman filter estimating level and slope (constant velocity)
    n = len(prices)
    # state: [price, velocity]
    x = np.zeros((2,))
    P = np.eye(2) * 0.1
    F = np.array([[1.0, 1.0],[0.0, 1.0]])
    Q = np.eye(2) * q
    H = np.array([1.0, 0.0]).reshape(1,2)
    R = np.array([[r]])
    estimates = []
    for z in prices:
        # predict
        x = F.dot(x)
        P = F.dot(P).dot(F.T) + Q
        # update
        y = np.array([[z]]) - H.dot(x).reshape(1,1)
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(np.linalg.inv(S))
        x = x + (K.dot(y)).flatten()
        P = (np.eye(2) - K.dot(H)).dot(P)
        estimates.append(x.copy())
    # return last predicted price next-step: level + velocity
    last = estimates[-1]
    pred = last[0] + last[1]
    return float(pred), estimates

# ---------------------
# Model builders / trainers
# ---------------------
def train_xgboost(X_train, y_train):
    if HAS_XGBOOST:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.XGBRegressor(**XGB_PARAMS, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        return model
    else:
        model = GradientBoostingRegressor(**GB_PARAMS, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        return model

def train_gbm(X_train, y_train):
    model = GradientBoostingRegressor(**GB_PARAMS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

# ---------------------
# Deep models (GRU / Transformer / Autoencoder)
# ---------------------
def build_gru_model(seq_len, n_features, units=GRU_PARAMS["units"]):
    inp = Input((seq_len, n_features))
    x = GRU(units, return_sequences=False)(inp)
    x = Dropout(0.1)(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

def prepare_sequences(df, feature_cols, seq_len):
    arr = df[feature_cols].values
    Xs, ys = [], []
    for i in range(len(arr) - seq_len - HORIZON + 1):
        Xs.append(arr[i:i+seq_len])
        ys.append(df["close"].values[i+seq_len+HORIZON-1])
    return np.array(Xs), np.array(ys)

def train_gru(df, feature_cols, seq_len=60, epochs=GRU_PARAMS["epochs"], batch=GRU_PARAMS["batch"]):
    if not HAS_TF:
        return None, 0.0
    X, y = prepare_sequences(df, feature_cols, seq_len)
    if len(X) < 100:
        return None, 0.0
    split = int(len(X)*(1-TEST_SIZE))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    model = build_gru_model(seq_len, X.shape[2])
    es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=epochs, batch_size=batch, callbacks=[es], verbose=0)
    # pseudo confidence: 1/(1+val_loss)
    val_loss = model.evaluate(X_val, y_val, verbose=0) if len(X_val) > 0 else 1.0
    conf = float(max(0.0, min(1.0, 1.0/(1.0+val_loss))))
    return model, conf

# tiny transformer encoder block for time series
def build_transformer_encoder(seq_len, n_features, d_model=TRANSFORMER_PARAMS["d_model"], num_heads=TRANSFORMER_PARAMS["num_heads"], ff_dim=TRANSFORMER_PARAMS["ff_dim"]):
    inp = Input((seq_len, n_features))
    x = Dense(d_model)(inp)
    att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(x, x)
    x = LayerNormalization()(att + x)
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(d_model)(ff)
    x = LayerNormalization()(ff + x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

def train_transformer(df, feature_cols, seq_len=60, epochs=TRANSFORMER_PARAMS["epochs"], batch=TRANSFORMER_PARAMS["batch"]):
    if not HAS_TF:
        return None, 0.0
    X, y = prepare_sequences(df, feature_cols, seq_len)
    if len(X) < 100:
        return None, 0.0
    split = int(len(X)*(1-TEST_SIZE))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    model = build_transformer_encoder(seq_len, X.shape[2])
    es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=epochs, batch_size=batch, callbacks=[es], verbose=0)
    val_loss = model.evaluate(X_val, y_val, verbose=0) if len(X_val) > 0 else 1.0
    conf = float(max(0.0, min(1.0, 1.0/(1.0+val_loss))))
    return model, conf

# Autoencoder for feature compression
def train_autoencoder(X, encoding_dim=AE_PARAMS["encoding_dim"], epochs=AE_PARAMS["epochs"], batch=AE_PARAMS["batch"]):
    if not HAS_TF:
        return None, None
    X = X.astype(np.float32)
    n_features = X.shape[1]
    inp = Input((n_features,))
    encoded = Dense(encoding_dim, activation="relu")(inp)
    decoded = Dense(n_features, activation="linear")(encoded)
    ae = Model(inp, decoded)
    ae.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="loss", patience=4, restore_best_weights=True)
    ae.fit(X, X, epochs=epochs, batch_size=batch, callbacks=[es], verbose=0)
    encoder = Model(inp, encoded)
    encoded_X = encoder.predict(X, verbose=0)
    return encoder, encoded_X

# ---------------------
# Helper: fit stacking ensemble (sklearn) or weighted blend
# ---------------------
def build_stacking_and_predict(X_train, y_train, X_test, estimators=None, final_estimator=None):
    # estimators: list of (name, estimator)
    if estimators is None:
        estimators = [("gb", GradientBoostingRegressor(**GB_PARAMS)),
                      ("rf", RandomForestRegressor(n_estimators=100))]
    if final_estimator is None:
        final_estimator = Ridge()
    try:
        stack = StackingRegressor(estimators=estimators, final_estimator=final_estimator, n_jobs=-1, passthrough=False)
        stack.fit(X_train, y_train)
        preds = stack.predict(X_test)
        # pseudo confidence: r2 on training (not ideal but quick)
        conf = float(max(0.0, min(1.0, r2_score(y_train, stack.predict(X_train)) if len(y_train)>10 else 0.5)))
        return stack, preds, conf
    except Exception as e:
        # fallback: train gbm and average
        gb = GradientBoostingRegressor(**GB_PARAMS)
        gb.fit(X_train, y_train)
        preds = gb.predict(X_test)
        conf = float(max(0.0, min(1.0, r2_score(y_train, gb.predict(X_train)) if len(y_train)>10 else 0.5)))
        return gb, preds, conf

# ---------------------
# Full pipeline
# ---------------------
def run_pipeline():
    print("Fetching data...")
    df = fetch_ohlcv(EXCHANGE_ID, SYMBOL, TIMEFRAME, LIMIT)
    df_feat = add_features(df)
    X, y, df2 = build_regression_dataset(df_feat, horizon=HORIZON)
    # keep index alignment for final latest sample
    X_idx = df2.index

    # split
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=TEST_SIZE, shuffle=False)
    # scaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # -----------------------
    # Kalman trend
    # -----------------------
    kalman_pred, _ = simple_kalman_trend(df2["close"].values)
    # pseudo confidence for kalman: low if noisy
    kalman_conf = 0.5

    # -----------------------
    # XGBoost / GBM
    # -----------------------
    print("Training XGBoost/GBM...")
    xgb_model = train_xgboost(X_train_s, y_train)
    xgb_pred = float(xgb_model.predict(X_test_s[-1].reshape(1,-1))[0])
    xgb_conf = 0.5
    try:
        # use r2 on test slice as proxy
        xgb_conf = float(max(0.0, min(1.0, r2_score(y_test, xgb_model.predict(X_test_s)))))
    except Exception:
        pass

    # -----------------------
    # Stacking ensemble (sklearn)
    # -----------------------
    print("Training stacking ensemble (sklearn) ...")
    estimators = []
    estimators.append(("gb", GradientBoostingRegressor(**GB_PARAMS)))
    estimators.append(("rf", RandomForestRegressor(n_estimators=100)))
    # train stacking on scaled features
    stack_model, stack_preds, stack_conf = build_stacking_and_predict(X_train_s, y_train, X_test_s, estimators=estimators, final_estimator=Ridge())
    stack_pred = float(stack_preds[-1]) if isinstance(stack_preds, np.ndarray) else float(stack_preds)

    # -----------------------
    # Autoencoder compression + retrain simple regressor on encoded features
    # -----------------------
    ae_pred = None; ae_conf = 0.0
    if HAS_TF:
        print("Training Autoencoder for feature compression...")
        encoder, encoded_train = train_autoencoder(X_train_s, encoding_dim=AE_PARAMS["encoding_dim"], epochs=AE_PARAMS["epochs"], batch=AE_PARAMS["batch"])
        if encoder is not None:
            encoded_test = encoder.predict(X_test_s, verbose=0)
            # fit simple model on encoded features
            reg = Ridge()
            reg.fit(encoded_train, y_train)
            ae_pred = float(reg.predict(encoded_test[-1].reshape(1,-1))[0])
            ae_conf = float(max(0.0, min(1.0, r2_score(y_train, reg.predict(encoded_train)) if len(y_train)>10 else 0.4)))

    # -----------------------
    # GRU
    # -----------------------
    gru_pred = None; gru_conf = 0.0
    feature_cols = list(X.columns)
    if HAS_TF:
        print("Training GRU...")
        gru_model, gru_conf = train_gru(df2[feature_cols + ["close"]], feature_cols, seq_len=60, epochs=GRU_PARAMS["epochs"], batch=GRU_PARAMS["batch"])
        if gru_model is not None:
            # prepare last sequence
            seq = df2[feature_cols].values[-60:].reshape(1,60,len(feature_cols))
            gru_pred = float(gru_model.predict(seq, verbose=0)[0,0])

    # -----------------------
    # Transformer
    # -----------------------
    trans_pred = None; trans_conf = 0.0
    if HAS_TF:
        print("Training Transformer encoder...")
        trans_model, trans_conf = train_transformer(df2[feature_cols + ["close"]], feature_cols, seq_len=60, epochs=TRANSFORMER_PARAMS["epochs"], batch=TRANSFORMER_PARAMS["batch"])
        if trans_model is not None:
            seq = df2[feature_cols].values[-60:].reshape(1,60,len(feature_cols))
            trans_pred = float(trans_model.predict(seq, verbose=0)[0,0])

    # -----------------------
    # Prepare model preds & confidences
    # -----------------------
    model_preds = {}
    model_confs = {}
    model_preds["kalman"] = kalman_pred; model_confs["kalman"] = kalman_conf
    model_preds["xgb"] = xgb_pred; model_confs["xgb"] = xgb_conf
    model_preds["stack"] = stack_pred; model_confs["stack"] = stack_conf
    if ae_pred is not None:
        model_preds["ae"] = ae_pred; model_confs["ae"] = ae_conf
    if gru_pred is not None:
        model_preds["gru"] = gru_pred; model_confs["gru"] = gru_conf
    if trans_pred is not None:
        model_preds["transformer"] = trans_pred; model_confs["transformer"] = trans_conf

    # -----------------------
    # Final blend: weighted by confidence
    # -----------------------
    preds = []
    confs = []
    for k,p in model_preds.items():
        preds.append(p)
        confs.append(model_confs.get(k,0.5))
    preds = np.array(preds)
    confs = np.array(confs)
    if confs.sum() == 0:
        weights = np.ones_like(confs)/len(confs)
    else:
        weights = confs / confs.sum()
    final_target = float(np.dot(weights, preds))

    # suggested SL/TP based on recent volatility (atr)
    recent_atr = float(df_feat["atr_14"].dropna().iloc[-1]) if "atr_14" in df_feat.columns else float(df2["tr"].rolling(14).mean().iloc[-1])
    latest_close = float(df2["close"].iloc[-1])
    # If final_target > price -> BUY idea, else SELL
    side = "HOLD"
    if final_target > latest_close * 1.001:
        side = "BUY"
        sl = latest_close - 2*recent_atr
        tp = final_target
    elif final_target < latest_close * 0.999:
        side = "SELL"
        sl = latest_close + 2*recent_atr
        tp = final_target
    else:
        sl = None; tp = None

    # -----------------------
    # Print tidy results
    # -----------------------
    def fmt(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:,.6f}"
        return str(x)

    print("\n=== Models predictions (next {} steps) ===".format(HORIZON))
    for k in model_preds:
        print(f"  {k:12s} -> pred: {fmt(model_preds[k])}  (conf: {fmt(model_confs.get(k,0.5))})")
    print(f"\nFinal blended target: {fmt(final_target)}  (side: {side})")
    print(f"Latest close: {fmt(latest_close)}  recent ATR: {fmt(recent_atr)}")
    print(f"Suggested Stop-Loss: {fmt(sl)}")
    print(f"Suggested Take-Profit: {fmt(tp)}")

    # return dictionary for programmatic access
    return {
        "models": model_preds,
        "confs": model_confs,
        "final_target": final_target,
        "side": side,
        "stop_loss": sl,
        "take_profit": tp,
        "latest_close": latest_close,
        "recent_atr": recent_atr
    }

# ---------------------
# Run
# ---------------------
if __name__ == "__main__":
    result = run_pipeline()

```   

## Training System performed and updated:    

```

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

# ----------------------------------------------------------
# Fetch OHLCV Bars
# ----------------------------------------------------------
def fetch_ohlcv_bars(exchange_id="binanceus", symbol="BTC/USDT", timeframe="4h", limit=1000):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

# ----------------------------------------------------------
# Compute VWAP + Std Bands
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------
def prepare_features(df):
    df = df.copy()
    
    df["return"] = df["close"].pct_change()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["volatility"] = df["return"].rolling(20).std()
    
    df["target"] = df["close"].shift(-1)   # Next-close

    df = df.dropna()  # Drop all rows that contain NaNs INCLUDING the last target row

    features = ["close", "ma20", "ma50", "volatility", "volume"]
    X = df[features]
    y = df["target"]

    return df, X, y

# ----------------------------------------------------------
# Ridge Regression Baseline
# ----------------------------------------------------------
def model_ridge(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    prediction = model.predict([Xs[-1]])
    return prediction[0]

# ----------------------------------------------------------
# XGBoost / Gradient Boosting
# ----------------------------------------------------------
def model_xgboost(X, y):
    model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5)
    model.fit(X, y)
    return model.predict([X.iloc[-1]])[0]

def model_gbm(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model.predict([X.iloc[-1]])[0]

# ----------------------------------------------------------
# Kalman Filter (Smoothed price)
# ----------------------------------------------------------
def apply_kalman(df):
    kf = KalmanFilter(
        initial_state_mean=df["close"].iloc[0],
        n_dim_obs=1,
        n_dim_state=1
    )

    smoothed, _ = kf.smooth(df["close"].values)

    return float(smoothed[-1])  # <--- FIX: ensure python float

# ----------------------------------------------------------
# Autoencoder for Latent Factors
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# GRU Model
# ----------------------------------------------------------
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
        layers.GRU(32, return_sequences=False),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)

    pred = model.predict(X[-1].reshape(1, seq, 1))[0][0]
    return pred

# ----------------------------------------------------------
# Transformer Model
# ----------------------------------------------------------
def model_transformer(df):
    seq = 30
    data = df["close"].values

    X, Y = [], []
    for i in range(len(data) - seq):
        X.append(data[i:i+seq])
        Y.append(data[i+seq])

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

# ----------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------
symbol = "BTC/USDT"
exchange_id = "binanceus"

bars = fetch_ohlcv_bars(exchange_id, symbol, timeframe="1w", limit=224)
summary = compute_vwap_std(bars)

df, X, y = prepare_features(bars)

# ML Predictions
ridge_pred = model_ridge(X, y)
xgb_pred = model_xgboost(X, y)
gbm_pred = model_gbm(X, y)
kalman_pred = apply_kalman(df)
ae_latent = model_autoencoder(X)
gru_pred = model_gru(df)
transformer_pred = model_transformer(df)

print("\n=== VWAP Summary ===")
for k, v in summary.items():
    print(f"{k:15}: {v}")

print("\n=== Machine Learning Forecasts ===")
print(f"Ridge Regression:     {ridge_pred:.2f}")
print(f"XGBoost:              {xgb_pred:.2f}")
print(f"GBM:                  {gbm_pred:.2f}")
print(f"Kalman Smoothed:      {kalman_pred:.2f}")
print(f"Autoencoder factors:  {ae_latent}")
print(f"GRU Forecast:         {gru_pred:.2f}")
print(f"Transformer Forecast: {transformer_pred:.2f}")


```  
