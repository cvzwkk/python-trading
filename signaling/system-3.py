
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
# MULTI-TIMEFRAME PIPELINE
# ==========================================================
from collections import Counter
from sklearn.metrics import mean_squared_error

def fetch_multi_timeframes(exchange_id, symbol, timeframes=["1h","4h","1d"], limit=500):
    """
    Busca OHLCV para cada timeframe e retorna dict de DataFrames.
    Usa sua função fetch_ohlcv_bars (que faz load_markets e fetch).
    """
    dfs = {}
    for tf in timeframes:
        try:
            df_tf = fetch_ohlcv_bars(exchange_id, symbol, timeframe=tf, limit=limit)
            dfs[tf] = df_tf
        except Exception as e:
            print(f"Erro ao buscar {tf}: {e}")
    return dfs

def prepare_last_features_for_tf(df, lookback=30):
    """
    Prepara features simples por timeframe: close, ma20, ma50, volatility e último preço.
    Retorna um pequeno dataframe com as últimas linhas úteis.
    """
    d = df.copy()
    d["return"] = d["close"].pct_change()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma50"] = d["close"].rolling(50).mean()
    d["volatility"] = d["return"].rolling(20).std()
    d = d.dropna()
    # garantir que haja dados
    if len(d) < lookback:
        raise ValueError("DataFrame muito curto para lookback")
    return d.iloc[-lookback:].reset_index(drop=True)

def predict_on_timeframe_models(df_tf):
    """
    Executa seu conjunto de modelos que aceitam 'df' e modelos que usam X,y.
    Retorna dicionário {model_name: prediction}
    NOTE: alguns modelos esperam X,y em vez de df (ex: model_deep_mlp, model_random_forest)
    """
    preds = {}
    # prepare tabular features
    df_feats, X, y = prepare_features(df_tf)
    last_close = df_feats["close"].iloc[-1]

    # Modelos que usam df directly
    try: preds["gru"] = float(model_gru(df_feats))
    except Exception as e: preds["gru"] = np.nan
    try: preds["transformer"] = float(model_transformer(df_feats))
    except Exception as e: preds["transformer"] = np.nan
    try: preds["cnn_lstm"] = float(model_cnn_lstm(df_feats))
    except Exception as e: preds["cnn_lstm"] = np.nan
    try: preds["bilstm"] = float(model_bilstm(df_feats))
    except Exception as e: preds["bilstm"] = np.nan
    try: preds["lstm_attn"] = float(model_lstm_attention(df_feats))
    except Exception as e: preds["lstm_attn"] = np.nan
    try: preds["tcn"] = float(model_tcn(df_feats))
    except Exception as e: preds["tcn"] = np.nan
    try: preds["cnn_gru"] = float(model_cnn_gru(df_feats))
    except Exception as e: preds["cnn_gru"] = np.nan
    try: preds["seq2seq"] = float(model_seq2seq(df_feats))
    except Exception as e: preds["seq2seq"] = np.nan

    # Tabular / classical models (X,y)
    try: preds["ridge"] = float(model_ridge(X, y))
    except Exception as e: preds["ridge"] = np.nan
    try: preds["xgb"] = float(model_xgboost(X, y))
    except Exception as e: preds["xgb"] = np.nan
    try: preds["gbm"] = float(model_gbm(X, y))
    except Exception as e: preds["gbm"] = np.nan
    try: preds["mlp"] = float(model_deep_mlp(X, y))
    except Exception as e: preds["mlp"] = np.nan
    try: preds["rf"] = float(model_random_forest(X, y))
    except Exception as e: preds["rf"] = np.nan

    # last_close for reference
    preds["last_close"] = last_close
    return preds

def stack_timeframe_features(dfs_dict):
    """
    Concatena features de cada timeframe em um vetor único para previsão stacked.
    Retorna um pandas.Series (1D) com valores ordenados por timeframe key.
    Exemplo de output: [close_1h, ma20_1h, vol_1h, close_4h, ma20_4h, ...]
    """
    vec = []
    keys = []
    for tf, df in dfs_dict.items():
        d = prepare_last_features_for_tf(df, lookback=60)  # pega 60 p/ mais robustez
        last = d.iloc[-1]
        vec.extend([last["close"], last["ma20"], last["ma50"], last["volatility"], last["volume"] if "volume" in last else 0])
        keys.append(tf)
    # construir pandas Series e manter ordem
    return pd.Series(vec, index=[f"{k}_{f}" for k in keys for f in ["close","ma20","ma50","volatility","volume"]])

def compute_simple_timeframe_weights(timeframes):
    """
    Heurística simples: quanto maior timeframe, maior estabilidade → maior peso.
    Ajuste conforme quiser. Retorna dict {tf: weight}
    """
    base = {}
    for tf in timeframes:
        # mapeamento simples: minutos extracted
        if tf.endswith("m"):
            mult = 1
        elif tf.endswith("h"):
            mult = int(tf[:-1]) * 60 if tf[:-1].isdigit() else 60
        elif tf.endswith("d"):
            mult = int(tf[:-1]) * 60 * 24 if tf[:-1].isdigit() else 1440
        else:
            mult = 60
        base[tf] = mult
    # normalizar
    s = sum(base.values())
    return {k: v/s for k,v in base.items()}

def aggregate_predictions_by_timeframe(all_preds, tf_weights=None, method="weighted_mean"):
    """
    all_preds: dict {tf: {model_name: pred, ...}}
    tf_weights: dict {tf: weight}
    method: "weighted_mean" or "voting" or "median"
    Retorna aggregated_pred (float) e votes (Counter)
    """
    # construir lista de best-model predictions por timeframe (poderíamos filtrar NaNs)
    best_preds = {}
    votes = []
    for tf, preds in all_preds.items():
        # filtrar nan e pegar média simples dos modelos daquele timeframe
        model_values = [v for k,v in preds.items() if k != "last_close" and not pd.isna(v)]
        if len(model_values) == 0:
            continue
        # escolha do representante: média dos modelos
        rep = float(np.mean(model_values))
        best_preds[tf] = rep
        # voto: BULL/BEAR/NEUTRAL segundo rep vs last_close
        last = preds.get("last_close", rep)
        if rep > last:
            votes.append("BULL")
        elif rep < last:
            votes.append("BEAR")
        else:
            votes.append("NEUTRAL")

    # agregado
    if tf_weights is None:
        # peso uniforme
        tf_weights = {tf: 1/len(best_preds) for tf in best_preds.keys()}
    # normalizar só para chocar com best_preds keys
    total = sum(tf_weights.get(tf,0) for tf in best_preds.keys())
    if total == 0:
        tf_weights = {tf: 1/len(best_preds) for tf in best_preds.keys()}
    else:
        tf_weights = {tf: tf_weights.get(tf,0)/total for tf in best_preds.keys()}

    if method == "weighted_mean":
        agg = sum(best_preds[tf] * tf_weights.get(tf,0) for tf in best_preds.keys())
    elif method == "median":
        agg = float(np.median(list(best_preds.values())))
    elif method == "voting":
        # majority vote between BULL/BEAR/NEUTRAL, then compute mean of timeframes that voted that
        cnt = Counter(votes)
        top = cnt.most_common(1)[0][0]
        sel_tfs = [tf for tf,p in best_preds.items() if ( "BULL" if best_preds[tf] > all_preds[tf].get("last_close",best_preds[tf]) else ("BEAR" if best_preds[tf] < all_preds[tf].get("last_close",best_preds[tf]) else "NEUTRAL")) == top]
        if len(sel_tfs)==0:
            agg = float(np.mean(list(best_preds.values())))
        else:
            agg = float(np.mean([best_preds[tf] for tf in sel_tfs]))
    else:
        agg = float(np.mean(list(best_preds.values())))

    return agg, Counter(votes)

# OPTIONAL: função que calcula pesos reais por timeframe via backtest rápido
def compute_timeframe_weights_historical(dfs_dict, model_func_list, test_frac=0.2):
    """
    Treina (rápido) modelos por timeframe e mede MSE em holdout;
    Retorna pesos inverso-MSE normalizados — mais baixo MSE => maior peso.
    model_func_list: lista de funções que recebem (X,y) ou df (tentar ambos)
    WARNING: pode ser lento; use com parcimônia.
    """
    tf_scores = {}
    for tf, df in dfs_dict.items():
        try:
            df_f, X, y = prepare_features(df)
            Xs = StandardScaler().fit_transform(X)
            X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=test_frac, shuffle=False)
            # usar Ridge como proxy
            m = Ridge(alpha=1.0)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            tf_scores[tf] = mse
        except Exception as e:
            tf_scores[tf] = np.inf
    # converter para pesos
    inv = {k: (1.0/(v+1e-9) if np.isfinite(v) else 0.0) for k,v in tf_scores.items()}
    s = sum(inv.values())
    if s == 0:
        return compute_simple_timeframe_weights(list(dfs_dict.keys()))
    return {k: v/s for k,v in inv.items()}

# ==========================================================
# Example usage: MULTI-TIMEFRAME EXECUTION
# ==========================================================
def forecast_single_timeframe(exchange_id, symbol, timeframe="1h", limit=800):
    try:
        bars = fetch_ohlcv_bars(exchange_id, symbol, timeframe=timeframe, limit=limit)
        df, X, y = prepare_features(bars)

        last_close = df["close"].iloc[-1]

        # ---- ML MODEL PREDICTIONS ----
        preds = {
            "Ridge": model_ridge(X, y),
            "XGBoost": model_xgboost(X, y),
            "GBM": model_gbm(X, y),
            "Kalman": apply_kalman(df),
            "GRU": model_gru(df),
            "Transformer": model_transformer(df),
            "CNN-LSTM": model_cnn_lstm(df),
            "BiLSTM": model_bilstm(df),
            "LSTM-Attention": model_lstm_attention(df),
            "TCN": model_tcn(df),
            "CNN-GRU": model_cnn_gru(df),
            "DeepMLP": model_mlp(df),
            "Seq2Seq": model_seq2seq(df),
            "RandomForest": model_rf(X, y)
        }

        # ---- SIGNALS ----
        signals = {k: price_signal(preds[k], last_close) for k in preds}

        # ---- FINAL SIGNAL: MAJORITY VOTE ----
        bullish = sum(1 for s in signals.values() if s == "BULLISH")
        bearish = sum(1 for s in signals.values() if s == "BEARISH")

        final_signal = (
            "BULLISH" if bullish > bearish else
            "BEARISH" if bearish > bullish else
            "NEUTRAL"
        )

        return {
            "timeframe": timeframe,
            "last_close": last_close,
            "predictions": preds,
            "signals": signals,
            "votes": {
                "bullish": bullish,
                "bearish": bearish,
                "neutral": len(preds) - bullish - bearish
            },
            "final_signal": final_signal
        }

    except Exception as e:
        print(f"Error on timeframe {timeframe}: {e}")
        return None

def multi_timeframe_forecast(exchange_id, symbol, timeframes, limit=1000):
    results = {}

    for tf in timeframes:
        r = forecast_single_timeframe(exchange_id, symbol, tf, limit)
        if r:
            results[tf] = r

    # GLOBAL VOTING
    total_bull = sum(r["votes"]["bullish"] for r in results.values())
    total_bear = sum(r["votes"]["bearish"] for r in results.values())

    if total_bull > total_bear:
        agg_signal = "BULLISH"
    elif total_bear > total_bull:
        agg_signal = "BEARISH"
    else:
        agg_signal = "NEUTRAL"

    return {
        "per_tf": results,
        "total_votes": {"bullish": total_bull, "bearish": total_bear},
        "aggregated_signal": agg_signal
    }


# ==========================================================
# Helper: interpret aggregated prediction
# ==========================================================
def interpret_agg(pred, last_close):
    if pred > last_close:
        return "BULLISH 📈"
    elif pred < last_close:
        return "BEARISH 📉"
    return "NEUTRAL ➖"

def price_signal(pred, last):
    if pred is None or last is None:
        return "N/A"
    if pred > last:
        return "BULLISH"
    if pred < last:
        return "BEARISH"
    return "NEUTRAL"

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

res = multi_timeframe_forecast(exchange_id, symbol, ["1h","4h","1d"], limit=900)

print("\n=== MULTI-TIMEFRAME FORECAST ===")
print("Aggregated Signal:", res["aggregated_signal"])
print("Total Votes:", res["total_votes"])

print("\n=== PER-TIMEFRAME ===")
for tf, r in res["per_tf"].items():
    print("\n-----------------------------------")
    print(f"Timeframe: {tf}")
    print(f"Last Close: {r['last_close']:.2f}")
    print("Final Signal:", r["final_signal"])
    print("Votes:", r["votes"])

    print("\nModel Predictions:")
    for model_name, pred in r["predictions"].items():
        pred_str = f"{pred:.2f}" if isinstance(pred, (int,float)) else "N/A"
        sig = r["signals"][model_name]
        print(f"  {model_name:<15}: {pred_str} → {sig}")
