
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ML Libraries
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models

# Kalman
from pykalman import KalmanFilter

# Prophet
from neuralprophet import NeuralProphet

# ARIMA
from pmdarima import auto_arima

# ==========================================================
# Fetch OHLCV Bars
# ==========================================================
def fetch_ohlcv_bars(exchange_id="binanceus", symbol="BTC/USDT", timeframe="4h", limit=1000):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

# ==========================================================
# VWAP + Std Bands
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
# Trend Signal
# ==========================================================
def trend_signal(pred, last_price):
    if pred > last_price:
        return "BULLISH 📈"
    elif pred < last_price:
        return "BEARISH 📉"
    return "NEUTRAL ➖"

# ==========================================================
# Ensemble
# ==========================================================
def ensemble_forecast(predictions, method="mean", weights=None):
    preds = np.array(list(predictions.values()))
    if method == "mean":
        return float(np.mean(preds))
    elif method == "median":
        return float(np.median(preds))
    elif method == "weighted":
        if weights is None:
            raise ValueError("Weights required for weighted ensemble")
        w = np.array([weights[k] for k in predictions.keys()])
        return float(np.sum(preds * w) / np.sum(w))
    else:
        raise ValueError("Invalid method")

# ==========================================================
# ================== MODELS ================================
# ==========================================================

# 1) Ridge Regression
def model_ridge(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(Xs, y, test_size=0.2)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return float(model.predict([Xs[-1]])[0])

# 2) XGBoost
def model_xgboost(X, y):
    model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5)
    model.fit(X, y)
    return float(model.predict([X.iloc[-1]])[0])

# 3) Gradient Boosting
def model_gbm(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return float(model.predict([X.iloc[-1]])[0])

# 4) Kalman
def model_kalman(df):
    kf = KalmanFilter(initial_state_mean=df["close"].iloc[0], n_dim_obs=1, n_dim_state=1)
    smoothed, _ = kf.smooth(df["close"].values)
    return float(smoothed[-1])

# 5) Autoencoder
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
    return float(latent_vec[0][0])

# 6) GRU
def model_gru(df):
    seq = 30
    data = df["close"].values
    X_seq, Y_seq = [], []
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq = np.array(X_seq).reshape(-1,seq,1)
    Y_seq = np.array(Y_seq)
    model = models.Sequential([layers.GRU(32), layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq, Y_seq, epochs=10, verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,seq,1))[0][0])

# 7) Transformer
def model_transformer(df):
    seq = 30
    data = df["close"].values
    X_seq, Y_seq = [], []
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq = np.array(X_seq).reshape(-1,seq,1)
    Y_seq = np.array(Y_seq)
    inp = layers.Input(shape=(seq,1))
    attn = layers.MultiHeadAttention(num_heads=2,key_dim=16)(inp,inp)
    x = layers.LayerNormalization()(inp+attn)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp,out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,seq,1))[0][0])

# 8) CNN-LSTM
def model_cnn_lstm(df):
    seq = 30
    data = df["close"].values
    X_seq, Y_seq = [], []
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq = np.array(X_seq).reshape(-1,seq,1)
    Y_seq = np.array(Y_seq)
    model = models.Sequential([
        layers.Conv1D(32,3,activation='relu',padding='causal',input_shape=(seq,1)),
        layers.Conv1D(32,3,activation='relu',padding='causal'),
        layers.MaxPooling1D(2),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,seq,1))[0][0])

# 9) BiLSTM
def model_bilstm(df):
    seq=30
    data=df["close"].values
    X_seq,Y_seq=[],[]
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq=np.array(X_seq).reshape(-1,seq,1)
    Y_seq=np.array(Y_seq)
    model=models.Sequential([layers.Bidirectional(layers.LSTM(32)),layers.Dense(1)])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,seq,1))[0][0])

# 10) LSTM + Attention
def model_lstm_attention(df):
    seq=30
    data=df["close"].values
    X_seq,Y_seq=[],[]
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq=np.array(X_seq).reshape(-1,seq,1)
    Y_seq=np.array(Y_seq)
    inp=layers.Input(shape=(seq,1))
    lstm_out=layers.LSTM(32,return_sequences=True)(inp)
    attn=layers.MultiHeadAttention(num_heads=2,key_dim=16)(lstm_out,lstm_out)
    x=layers.GlobalAveragePooling1D()(attn)
    out=layers.Dense(1)(x)
    model=models.Model(inp,out)
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,seq,1))[0][0])

# 11) TCN
def model_tcn(df):
    seq=30
    data=df["close"].values
    X_seq,Y_seq=[],[]
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq=np.array(X_seq).reshape(-1,seq,1)
    Y_seq=np.array(Y_seq)
    model=models.Sequential([
        layers.Conv1D(64,3,dilation_rate=1,padding='causal',activation='relu',input_shape=(seq,1)),
        layers.Conv1D(64,3,dilation_rate=2,padding='causal',activation='relu'),
        layers.Conv1D(64,3,dilation_rate=4,padding='causal',activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,seq,1))[0][0])

# 12) CNN-GRU
def model_cnn_gru(df):
    seq=30
    data=df["close"].values
    X_seq,Y_seq=[],[]
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq=np.array(X_seq).reshape(-1,seq,1)
    Y_seq=np.array(Y_seq)
    model=models.Sequential([
        layers.Conv1D(32,3,activation='relu',padding='causal',input_shape=(seq,1)),
        layers.Conv1D(32,3,activation='relu',padding='causal'),
        layers.MaxPooling1D(2),
        layers.GRU(32),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,seq,1))[0][0])

# 13) MLP Deep
def model_deep_mlp(X,y):
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    model=models.Sequential([
        layers.Dense(64,activation='relu'),
        layers.Dense(64,activation='relu'),
        layers.Dense(32,activation='relu'),
        layers.Dense(16,activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam',loss='mse')
    model.fit(Xs,y,epochs=20,verbose=0)
    return float(model.predict(Xs[-1].reshape(1,-1))[0][0])

# 14) Seq2Seq
def model_seq2seq(df):
    seq=30
    data=df["close"].values
    X_seq,Y_seq=[],[]
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq=np.array(X_seq).reshape(-1,seq,1)
    Y_seq=np.array(Y_seq)
    inp=layers.Input(shape=(seq,1))
    enc=layers.LSTM(32,return_state=True)
    enc_out,h,c=enc(inp)
    dec_inp=layers.RepeatVector(1)(enc_out)
    dec_out=layers.LSTM(32)(dec_inp)
    out=layers.Dense(1)(dec_out)
    model=models.Model(inp,out)
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=10,verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,seq,1))[0][0])

# 15) Random Forest
def model_random_forest(X,y):
    model=RandomForestRegressor(n_estimators=300)
    model.fit(X,y)
    return float(model.predict([X.iloc[-1]])[0])

# 16) N-BEATS
def model_nbeats(df):
    seq=30
    data=df["close"].values
    X_seq,Y_seq=[],[]
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq=np.array(X_seq)
    Y_seq=np.array(Y_seq)
    inp=layers.Input(shape=(seq,))
    b1=layers.Dense(256,activation='relu')(inp)
    b1=layers.Dense(256,activation='relu')(b1)
    forecast=layers.Dense(1)(b1)
    model=models.Model(inp,forecast)
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=15,verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,-1))[0][0])

# 17) DeepAR
def model_deepar(df):
    seq=30
    data=df["close"].values
    X_seq,Y_seq=[],[]
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq=np.array(X_seq).reshape(-1,seq,1)
    Y_seq=np.array(Y_seq)
    model=models.Sequential([
        layers.LSTM(40,return_sequences=True),
        layers.LSTM(20),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=15,verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,seq,1))[0][0])

# 18) WaveNet
def model_wavenet(df):
    seq=30
    data=df["close"].values
    X_seq,Y_seq=[],[]
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq=np.array(X_seq).reshape(-1,seq,1)
    Y_seq=np.array(Y_seq)
    inp=layers.Input(shape=(seq,1))
    x=inp
    for d in [1,2,4,8,16]:
        x=layers.Conv1D(32,2,dilation_rate=d,padding='causal',activation='relu')(x)
    x=layers.GlobalAveragePooling1D()(x)
    out=layers.Dense(1)(x)
    model=models.Model(inp,out)
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=12,verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,seq,1))[0][0])

# 19) Informer
def model_informer(df):
    seq=30
    data=df["close"].values
    X_seq,Y_seq=[],[]
    for i in range(len(data)-seq):
        X_seq.append(data[i:i+seq])
        Y_seq.append(data[i+seq])
    X_seq=np.array(X_seq).reshape(-1,seq,1)
    Y_seq=np.array(Y_seq)
    inp=layers.Input(shape=(seq,1))
    x=layers.Conv1D(32,3,padding='causal',activation='relu')(inp)
    x=layers.MultiHeadAttention(num_heads=4,key_dim=16)(x,x)
    x=layers.GlobalAveragePooling1D()(x)
    out=layers.Dense(1)(x)
    model=models.Model(inp,out)
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_seq,Y_seq,epochs=12,verbose=0)
    return float(model.predict(X_seq[-1].reshape(1,seq,1))[0][0])

# 20) Prophet-Neural
def model_prophet_neural(df):
    df2=df.copy()
    df2["ds"]=df2.index
    df2["y"]=df2["close"]
    m=Prophet()
    m.fit(df2)
    future=m.make_future_dataframe(periods=1,freq='H')
    forecast=m.predict(future)
    return float(forecast["yhat"].iloc[-1])

# 21) CatBoost
def model_catboost(X,y):
    model=CatBoostRegressor(verbose=0)
    model.fit(X,y)
    return float(model.predict([X.iloc[-1]])[0])

# 22) LightGBM
def model_lightgbm(X,y):
    model=lgb.LGBMRegressor()
    model.fit(X,y)
    return float(model.predict([X.iloc[-1]])[0])

# 23) XGBoost Deep
def model_xgboost_deep(X,y):
    model=XGBRegressor(n_estimators=500,max_depth=7,learning_rate=0.02)
    model.fit(X,y)
    return float(model.predict([X.iloc[-1]])[0])

# 24) ElasticNet
def model_elasticnet(X,y):
    model=ElasticNet(alpha=0.1)
    model.fit(X,y)
    return float(model.predict([X.iloc[-1]])[0])

# 25) ARIMA
def model_arima(df):
    model=auto_arima(df["close"],seasonal=False)
    pred=model.predict(n_periods=1)
    return float(pred[0])

# 26) Placeholder for custom / future model
def model_custom(df):
    return float(df["close"].iloc[-1]) # apenas o último valor como fallback

# ==========================================================
# ================= MAIN EXECUTION ========================
# ==========================================================
symbol="BTC/USDT"
exchange_id="binanceus"
bars=fetch_ohlcv_bars(exchange_id,symbol,timeframe="4h",limit=224)
summary=compute_vwap_std(bars)
df,X,y=prepare_features(bars)
last_close=df["close"].iloc[-1]

# ==========================================================
# Obter previsões de todos os modelos
# ==========================================================
all_predictions = {
    "ridge": model_ridge(X,y),
    "xgboost": model_xgboost(X,y),
    "gbm": model_gbm(X,y),
    "kalman": model_kalman(df),
    "autoencoder": model_autoencoder(X),
    "gru": model_gru(df),
    "transformer": model_transformer(df),
    "cnn_lstm": model_cnn_lstm(df),
    "bilstm": model_bilstm(df),
    "lstm_attention": model_lstm_attention(df),
    "tcn": model_tcn(df),
    "cnn_gru": model_cnn_gru(df),
    "deep_mlp": model_deep_mlp(X,y),
    "seq2seq": model_seq2seq(df),
    "random_forest": model_random_forest(X,y),
    "nbeats": model_nbeats(df),
    "deepar": model_deepar(df),
    "wavenet": model_wavenet(df),
    "informer": model_informer(df),
    "prophet_neural": model_prophet_neural(df),
    "catboost": model_catboost(X,y),
    "lightgbm": model_lightgbm(X,y),
    "xgboost_deep": model_xgboost_deep(X,y),
    "elasticnet": model_elasticnet(X,y),
    "arima": model_arima(df),
    "custom": model_custom(df)
}

# Ensemble médio
ensemble_pred = ensemble_forecast(all_predictions,method="mean")
ensemble_signal = trend_signal(ensemble_pred,last_close)

# Ensemble ponderado (exemplo de pesos)
weights={k:1.0 for k in all_predictions.keys()} # pesos iguais para demo
ensemble_weighted_pred = ensemble_forecast(all_predictions,method="weighted",weights=weights)
ensemble_weighted_signal = trend_signal(ensemble_weighted_pred,last_close)

# ==========================================================
# PRINT OUTPUT
# ==========================================================
print("\n=== VWAP Summary ===")
for k,v in summary.items():
    print(f"{k:15}: {v}")

print("\n=== ENSEMBLE FORECAST ===")
print(f"Current Close: {last_close:.2f}")
print(f"Ensemble Prediction (Mean)   : {ensemble_pred:.2f} → {ensemble_signal}")
print(f"Ensemble Prediction (Weighted): {ensemble_weighted_pred:.2f} → {ensemble_weighted_signal}")
