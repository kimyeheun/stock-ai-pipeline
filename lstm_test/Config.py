class TestConfig:
    MODEL_PATH = "./models/lstm_classifier.pt"
    SCALER_PATH = "./models/scaler.pkl"
    FILE_PATH = "./raw_data/stock_data_from_api_kakao.csv"

    PRICE_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
    INDICATOR_FEATURES = ["RSI", "MACD", "MACD_SIGNAL", "BB_UPPER", "BB_LOWER", "MOM", "CCI"]
    ALL_FEATURES = PRICE_FEATURES + INDICATOR_FEATURES