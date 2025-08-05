
class TrainingConfigBefore:
    X_PATH = "./data/X_stock.npy"
    Y_PATH = "./data/y_stock.npy"

    SCALER_PATH = "models/demo/scaler_masked.pkl"
    MODEL_PATH = "models/demo/lstm_classifier.pt"
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    CLIP_GRAD_NORM = 1.0
    TEST_SIZE = 0.2
    SHUFFLE_DATA = True

    ALL_FEATURES = ["Open", "High", "Low", "Close", "Volume",
                    "RSI", "MACD", "MACD_SIGNAL", "BB_UPPER", "BB_LOWER", "MOM", "CCI"]
    # SELECTED_FEATURES = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "BB_UPPER", "BB_LOWER", "MOM", "CCI"]


class TrainingConfig:
    # BEFORE
    X_PATH = "../data_collection/data/X_stock.npy"
    Y_PATH = "../data_collection/data/y_stock.npy"
    SCALER_PATH = "../data_collection/models/scaler.pkl"
    # AFTER
    SPLIT_PATH = "../data_collection/data/split_data"
    MODEL_PATH = "models/lstm_classifier.pt"

    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    CLIP_GRAD_NORM = 1.0
    TEST_SIZE = 0.2
    SHUFFLE_DATA = True

    PRICE_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
    INDICATOR_FEATURES = ["RSI", "MACD", "MACD_SIGNAL", "BB_UPPER", "BB_LOWER", "MOM", "CCI"]
    ALL_FEATURES = PRICE_FEATURES + INDICATOR_FEATURES