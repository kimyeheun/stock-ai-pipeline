import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreProcessingConfig:
    RAW_DATA_DIR = "./raw_data"
    DATA_DIR= "./data"
    MODELS_DIR = "./models"
    STOCK_DATA_CSV = os.path.join(RAW_DATA_DIR, "stock_data_from_api_kakao.csv")
    SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
    X_PATH = os.path.join(DATA_DIR, "X_stock.npy")
    Y_PATH = os.path.join(DATA_DIR, "y_stock.npy")
    COLUMN_MAPPING = {
        'mkp': 'Open',
        'hipr': 'High',
        'lopr': 'Low',
        'clpr': 'Close',
        'trqu': 'Volume'
    }
    FEATURES = ["Open", "High", "Low", "Close", "Volume",
                "RSI", "MACD", "MACD_SIGNAL", "BB_UPPER", "BB_LOWER", "MOM", "CCI"]
    WINDOW_SIZE = 30
    LABEL_FUTURE_WINDOW = 10
    LABEL_THRESHOLD = 0.05

def load_stock_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower() for col in df.columns]
    df.rename(columns=DataPreProcessingConfig.COLUMN_MAPPING, inplace=True)
    df = df.dropna(subset=['Close'])
    return df

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def compute_bollinger_bands(close: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series]:
    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return upper, lower

def compute_mom(close: pd.Series, period: int = 10) -> pd.Series:
    mom = close.diff(period)
    return mom

def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical_price = (high + low + close) / 3
    sma_typical_price = typical_price.rolling(window=period).mean()
    def calculate_mean_deviation(series):
        if len(series) < period:
            return np.nan
        sma = series.mean()
        return np.mean(np.abs(series - sma))
    mean_deviation = typical_price.rolling(window=period).apply(calculate_mean_deviation, raw=False)
    denominator = 0.015 * mean_deviation
    cci = (typical_price - sma_typical_price) / denominator
    return cci

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['RSI'] = compute_rsi(df['Close'])
    macd, macd_signal = compute_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_SIGNAL'] = macd_signal
    bb_upper, bb_lower = compute_bollinger_bands(df['Close'])
    df['BB_UPPER'] = bb_upper
    df['BB_LOWER'] = bb_lower
    df['MOM'] = compute_mom(df['Close'])
    df['CCI'] = compute_cci(df['High'], df['Low'], df['Close'])
    return df

def create_buy_sell_hold_labels(df: pd.DataFrame, future_window: int = 10, threshold: float = 0.05) -> pd.DataFrame:
    labels = []
    close = df['Close'].values
    for i in range(len(df)):
        if i + future_window >= len(df):
            labels.append(np.nan)
            continue
        future_max = np.max(close[i + 1:i + future_window + 1])
        future_min = np.min(close[i + 1:i + future_window + 1])
        pct_increase = (future_max - close[i]) / close[i]
        pct_decrease = (future_min - close[i]) / close[i]
        if pct_increase >= threshold:
            labels.append(1)
        elif pct_decrease <= -threshold:
            labels.append(2)
        else:
            labels.append(0)
    df['Label'] = labels
    df = df.dropna(subset=['Label'])
    df.loc[:, 'Label'] = df['Label'].astype(int)
    return df

def prepare_sequences(df: pd.DataFrame, features: list[str], window_size: int = 30) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    data = df[features].values
    labels = df['Label'].values
    for i in range(window_size, len(df)):
        seq_x = data[i-window_size:i]
        seq_y = labels[i]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def scale_features_and_save(X: np.ndarray, y: np.ndarray, scaler_path: str, X_path: str, y_path: str):
    N, T, F = X.shape
    scaler = StandardScaler()
    X_reshape = X.reshape(-1, F)
    X_scaled = scaler.fit_transform(X_reshape)
    X_scaled = X_scaled.reshape(N, T, F)
    np.save(X_path, X_scaled)
    np.save(y_path, y)
    joblib.dump(scaler, scaler_path)
    print(f"Saved: {scaler_path}, {X_path}, {y_path}")


if __name__ == "__main__":
    config = DataPreProcessingConfig

    # 1. Load data
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    df = load_stock_data(config.STOCK_DATA_CSV)

    # 2. Compute indicators
    df = add_technical_indicators(df)

    # 3. Create labels
    df = create_buy_sell_hold_labels(df,
                                     future_window=config.LABEL_FUTURE_WINDOW,
                                     threshold=config.LABEL_THRESHOLD)

    # 4. Prepare sequential features and labels
    features = config.FEATURES
    window_size = config.WINDOW_SIZE
    df[features] = df[features].fillna(0)
    X, y = prepare_sequences(df, features, window_size=window_size)

    # 5. Preprocess and save result
    scale_features_and_save(X, y, config.SCALER_PATH, config.X_PATH, config.Y_PATH)

    print("Stock result preprocessing pipeline completed.")
