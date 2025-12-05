
#!pip install tensorflow-addons
!pip install neuralprophet
!pip install pmdarima
!pip install catboost
!pip install ccxt
!pip install mplfinance
#!pip install prophet==1.1.5 cmdstanpy==1.2.1
!pip install pykalman

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from pykalman import KalmanFilter
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras import layers, models

# ===================== UTILITÁRIOS =====================
def safe_float(x):
    if isinstance(x, np.ndarray):
        return x.item()
    return float(x)

def trend_signal(pred, last_price):
    if pred > last_price: return "BULLISH 📈"
    elif pred < last_price: return "BEARISH 📉"
    return "NEUTRAL ➖"

# ===================== FETCH ===========================
def fetch_ohlcv_bars(exchange_id="binanceus", symbol="BTC/USDT", timeframe="4h", limit=500):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

# ===================== FEATURE ENGINEERING ============
def prepare_features(df):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["volatility"] = df["return"].rolling(20).std()
    df["target"] = df["close"].shift(-1)
    df = df.dropna()
    features = ["close","ma20","ma50","volatility","volume"]
    X = df[features].values
    y = df["target"].values
    return df, X, y

# ===================== VWAP ============================
def compute_vwap_std(df):
    tp = (df["high"] + df["low"] + df["close"])/3
    vol = df["volume"].fillna(0)
    vwap = (tp*vol).sum()/vol.sum()
    std1 = np.sqrt(((tp-vwap)**2).mean())
    std2 = std1*2
    return {
        "VWAP": vwap,
        "Upper 1σ": vwap+std1,
        "Upper 2σ": vwap+std2,
        "Lower 1σ": vwap-std1,
        "Lower 2σ": vwap-std2,
        "Latest datetime": df["datetime"].iloc[-1],
        "Latest close": df["close"].iloc[-1]
    }

# ===================== KALMAN ==========================
def model_kalman(df):
    kf = KalmanFilter(initial_state_mean=df["close"].iloc[0], n_dim_obs=1, n_dim_state=1)
    smoothed,_ = kf.smooth(df["close"].values)
    return safe_float(smoothed[-1])

# ===================== ML MODELS =======================
def model_ridge(X,y): return safe_float(Ridge(alpha=1.0).fit(X,y).predict(X[-1].reshape(1,-1)))
def model_elasticnet(X,y): return safe_float(ElasticNet(alpha=0.1).fit(X,y).predict(X[-1].reshape(1,-1)))
def model_xgboost(X,y): return safe_float(XGBRegressor(n_estimators=300,learning_rate=0.03,max_depth=5).fit(X,y).predict(X[-1].reshape(1,-1)))
def model_gbm(X,y): return safe_float(GradientBoostingRegressor().fit(X,y).predict(X[-1].reshape(1,-1)))
def model_rf(X,y): return safe_float(RandomForestRegressor(n_estimators=200).fit(X,y).predict(X[-1].reshape(1,-1)))
def model_catboost(X,y): return safe_float(CatBoostRegressor(verbose=0).fit(X,y).predict(X[-1].reshape(1,-1)))
def model_lightgbm(X,y): return safe_float(lgb.LGBMRegressor().fit(X,y).predict(X[-1].reshape(1,-1)))

# ===================== DEEP LEARNING ===================
def build_seq(df, seq_len=30):
    data = df["close"].values
    X_seq,Y_seq = [],[]
    for i in range(len(data)-seq_len):
        X_seq.append(data[i:i+seq_len])
        Y_seq.append(data[i+seq_len])
    X_seq = np.array(X_seq).reshape(-1,seq_len,1)
    Y_seq = np.array(Y_seq)
    return X_seq,Y_seq

def model_gru(df):
    X_seq,Y_seq = build_seq(df)
    model = models.Sequential([layers.GRU(32,input_shape=(X_seq.shape[1],1)), layers.Dense(1)])
    model.compile(optimizer="adam",loss="mse")
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return safe_float(model.predict(X_seq[-1].reshape(1,X_seq.shape[1],1))[0][0])

def model_lstm(df):
    X_seq,Y_seq = build_seq(df)
    model = models.Sequential([layers.LSTM(32,input_shape=(X_seq.shape[1],1)), layers.Dense(1)])
    model.compile(optimizer="adam",loss="mse")
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return safe_float(model.predict(X_seq[-1].reshape(1,X_seq.shape[1],1))[0][0])

def model_bilstm(df):
    X_seq,Y_seq = build_seq(df)
    model = models.Sequential([layers.Bidirectional(layers.LSTM(32)), layers.Dense(1)])
    model.compile(optimizer="adam",loss="mse")
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return safe_float(model.predict(X_seq[-1].reshape(1,X_seq.shape[1],1))[0][0])

def model_cnn_lstm(df):
    X_seq,Y_seq = build_seq(df)
    model = models.Sequential([
        layers.Conv1D(32,3,activation='relu',input_shape=(X_seq.shape[1],1)),
        layers.MaxPooling1D(2),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return safe_float(model.predict(X_seq[-1].reshape(1,X_seq.shape[1],1))[0][0])

def model_cnn_gru(df):
    X_seq,Y_seq = build_seq(df)
    model = models.Sequential([
        layers.Conv1D(32,3,activation='relu',input_shape=(X_seq.shape[1],1)),
        layers.MaxPooling1D(2),
        layers.GRU(32),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return safe_float(model.predict(X_seq[-1].reshape(1,X_seq.shape[1],1))[0][0])

# ===================== ARIMA ===========================
def model_arima(df):
    model = ARIMA(df["close"].values,order=(5,1,0))
    model_fit = model.fit()
    return safe_float(model_fit.forecast()[0])

# ===================== ROLLING ML MODELS ==============
def model_ridge_rolling(df, window=50):
    vals = df['close'].values[-window:]
    X = vals[:-1].reshape(-1,1)
    y = vals[1:]
    return float(Ridge().fit(X,y).predict(X[-1].reshape(1,-1))[0])

def model_elasticnet_rolling(df, window=50):
    vals = df['close'].values[-window:]
    X = vals[:-1].reshape(-1,1)
    y = vals[1:]
    return float(ElasticNet().fit(X,y).predict(X[-1].reshape(1,-1))[0])

def model_xgb_rolling(df, window=50):
    vals = df['close'].values[-window:]
    X = vals[:-1].reshape(-1,1)
    y = vals[1:]
    return float(XGBRegressor().fit(X,y).predict(X[-1].reshape(1,-1))[0])

def model_lgb_rolling(df, window=50):
    vals = df['close'].values[-window:]
    X = vals[:-1].reshape(-1,1)
    y = vals[1:]
    return float(lgb.LGBMRegressor().fit(X,y).predict(X[-1].reshape(1,-1))[0])

# ===================== ENSEMBLE ========================
def ensemble_forecast(pred_dict, method="mean"):
    preds = list(pred_dict.values())
    if method=="mean":
        return np.mean(preds)
    elif method=="median":
        return np.median(preds)
    return preds[-1]

# ===================== MAIN ============================
symbol = "BTC/USDT"
exchange_id = "binanceus"

df = fetch_ohlcv_bars(exchange_id,symbol,timeframe="4h",limit=500)
summary = compute_vwap_std(df)
df, X, y = prepare_features(df)
last_close = df["close"].iloc[-1]

# ===================== PREDIÇÕES =======================
all_preds = {
    "ridge": model_ridge(X,y),
    "elasticnet": model_elasticnet(X,y),
    "xgboost": model_xgboost(X,y),
    "gbm": model_gbm(X,y),
    "rf": model_rf(X,y),
    "catboost": model_catboost(X,y),
    "lightgbm": model_lightgbm(X,y),
    "kalman": model_kalman(df),
    "gru": model_gru(df),
    "lstm": model_lstm(df),
    "bilstm": model_bilstm(df),
    "cnn_lstm": model_cnn_lstm(df),
    "cnn_gru": model_cnn_gru(df),
    "arima": model_arima(df),
    "ridge_roll": model_ridge_rolling(df),
    "elasticnet_roll": model_elasticnet_rolling(df),
    "xgb_roll": model_xgb_rolling(df),
    "lgb_roll": model_lgb_rolling(df)
}

ensemble_pred = ensemble_forecast(all_preds,method="mean")
ensemble_signal = trend_signal(ensemble_pred,last_close)

# ===================== OUTPUT ===========================
print("\n=== VWAP Summary ===")
for k,v in summary.items(): print(f"{k:15}: {v}")

print("\n=== Predictions ===")
print(f"Current Close: {last_close:.2f}")
for k,v in all_preds.items():
    print(f"{k:15}: {v:.2f} → {trend_signal(v,last_close)}")

print(f"\nEnsemble Forecast: {ensemble_pred:.2f} → {ensemble_signal}")
