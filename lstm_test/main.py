import time

import joblib

from Model import MaskAwareLSTM
from calc_indicator import add_technical_indicators, ensure_all_features_exist
from data_collection.Config import DataPreProcessingConfig
from strategies import *
from visualization import *


def load_stock_data(file_path: str):
    df = pd.read_csv(file_path)
    df.rename(columns=DataPreProcessingConfig.COLUMN_MAPPING, inplace=True)
    df = df.dropna(subset=['Close'])
    return df


if __name__ == '__main__':
    stock_data = load_stock_data(TestConfig.FILE_PATH)

    start = time.time()

    # TODO: 프롬프트로부터 indicators 뽑아내기
    use_indicators = ["RSI", "MACD", "MACD_SIGNAL", "BB_UPPER", "BB_LOWER", "MOM", "CCI"]
    indicators = add_technical_indicators(stock_data, use_indicators)
    stock_data = ensure_all_features_exist(stock_data, TestConfig.ALL_FEATURES)

    # MaskAwareLSTM의 인자: input_dim, hidden_dim, num_layers=2, dropout=0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskAwareLSTM(input_dim=len(TestConfig.ALL_FEATURES), hidden_dim=64, output_dim=3, num_layers=2, dropout=0.3).to(device)

    # state_dict와 scaler 경로 확인!
    state_dict = torch.load(TestConfig.MODEL_PATH, map_location=device)
    scaler = joblib.load(TestConfig.SCALER_PATH)
    model.load_state_dict(state_dict)

    # 초기화 및 실행
    rsi_strategy = UpperStrategy()
    rsi_result = rsi_strategy.run(stock_data)

    print("stock :", stock_data.columns , " " , stock_data.shape)
    # rsi_model_strategy = IntermediateStrategy()
    # rsi_model_result = rsi_model_strategy.run(stock_data, indicators, model, scaler,
    #                                           window_size=30)

    model_only_strategy = LowerStrategy()
    model_only_result = model_only_strategy.run(stock_data, model, scaler,
                                                window_size=30, indicators=use_indicators)

    create_comparison_chart_model(stock_data, stock_data.index, rsi_result, model_only_result)

    end = time.time()
    print("Total elapsed:", end - start)
