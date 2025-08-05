import os

class DataPreProcessingConfig:
    DATA_DIR= "./data"

    MODELS_DIR = "./models"
    SPLIT_PATH = os.path.join(DATA_DIR, "split_data")
    RAW_DATA_DIR = "./results"

    SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
    X_PATH = os.path.join(DATA_DIR, "X_stock.npy")
    Y_PATH = os.path.join(DATA_DIR, "y_stock.npy")

    STOCK_DATA_CSV = os.path.join(RAW_DATA_DIR, "stock_data_from_api_kakao.csv")

    COLUMN_MAPPING = {
        'mkp': 'Open',
        'hipr': 'High',
        'lopr': 'Low',
        'clpr': 'Close',
        'trqu': 'Volume'
    }

    ALL_FEATURES = ["Open", "High", "Low", "Close", "Volume",
                "RSI", "MACD", "MACD_SIGNAL", "BB_UPPER", "BB_LOWER", "MOM", "CCI"]
    WINDOW_SIZE = 30
    LABEL_FUTURE_WINDOW = 10
    LABEL_THRESHOLD = 0.05
