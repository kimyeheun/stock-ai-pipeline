import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib
import torch

from Model import MaskAwareLSTM
from train_model_ver2 import TrainingConfig


def load_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower() for col in df.columns]

    # clpr → close 매핑
    if 'clpr' in df.columns:
        df.rename(columns={'clpr': 'close'}, inplace=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        x_axis = df['date']
    else:
        x_axis = df.index

    stock_data = df
    # 컬럼명 매핑 (open/high/low/volume)
    stock_data['open'] = stock_data['mkp']
    stock_data['high'] = stock_data['hipr']
    stock_data['low'] = stock_data['lopr']
    stock_data['volume'] = stock_data['trqu']

    stock_data['rsi'] = talib.RSI(stock_data['close'], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(stock_data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    stock_data['macd'] = macd
    stock_data['macd_signal'] = macdsignal
    upper, middle, lower = talib.BBANDS(stock_data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    stock_data['bb_upper'] = upper
    stock_data['bb_lower'] = lower
    stock_data['mom'] = talib.MOM(stock_data['close'], timeperiod=10)
    stock_data['cci'] = talib.CCI(stock_data['high'], stock_data['low'], stock_data['close'], timeperiod=20)
    return x_axis, stock_data

def calc_talib(stock_data, indicators=None):
    if indicators is None:
        indicators = []
    result = {}
    if "rsi" in indicators:
        result["rsi"] = talib.RSI(stock_data["close"], timeperiod=14)
    return result

def strategy_rsi_rebound(stock_data, x_axis, indicators):
    portfolio = 100000
    cash = portfolio
    shares = 0
    in_trade = False

    buy_markers = []
    sell_markers = []

    rsi = indicators.get("rsi")
    trades = 0

    for i in range(1, len(stock_data)):
        price = stock_data["close"].iloc[i]
        rsi_prev = rsi.iloc[i-1]
        rsi_curr = rsi.iloc[i]

        if (not in_trade and pd.notna(rsi_prev) and pd.notna(rsi_curr)
                and rsi_prev < 30 and rsi_curr >= 30):
            shares = cash / price
            cash = 0
            in_trade = True
            trades += 1
            buy_markers.append(dict(
                x=x_axis[i],
                y=price,
                text=f"Buy: {price:.2f} (RSI 반등)",
                type='buy'
            ))
        elif in_trade and pd.notna(rsi_curr) and rsi_curr >= 70:
            cash = shares * price
            shares = 0
            in_trade = False
            trades += 1
            profit = cash - portfolio
            sell_markers.append(dict(
                x=x_axis[i],
                y=price,
                text=f"Sell: {price:.2f} (RSI>=70)<br>Profit: {profit:.2f}",
                type='sell'
            ))
    final_value = cash + shares * stock_data["close"].iloc[-1]
    profit_ratio = (final_value - portfolio) / portfolio
    return {
        "name": "RSI 반등(30) 매수, 70 매도 전략",
        "buy": buy_markers,
        "sell": sell_markers,
        "final_value": final_value,
        "num_trades": trades,
        "profit_ratio": profit_ratio
    }

def strategy_rsi_rebound_model_based(stock_data, x_axis, indicators, model, scaler, window_size=30):
    portfolio = 100000
    cash = portfolio
    shares = 0
    in_trade = False

    buy_markers = []
    sell_markers = []
    trades = 0

    selected_features = [f.lower() for f in TrainingConfig.ALL_FEATURES]
    valid_idx = []
    X = []
    mask = []

    for i in range(window_size, len(stock_data)):
        window = stock_data[selected_features].iloc[i - window_size:i].values
        X.append(window)
        m = (~np.isnan(window)).astype(np.float32)

        # 가격 피처는 항상 마스크 유지
        for feat in TrainingConfig.PRICE_FEATURES:
            idx = selected_features.index(feat.lower())
            m[:, idx] = 1.0
        mask.append(m)

        valid_idx.append(i)

    X = np.array(X)
    mask = np.array(mask)
    X = scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).to(device)
        masks = torch.tensor(mask, dtype=torch.float32).to(device)
        outputs = model(inputs, masks)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    rsi = indicators.get("rsi")

    for idx, i in enumerate(valid_idx[1:], 1):  # prev rsi 확인 위해 1부터 시작
        price = stock_data["close"].iloc[i]
        rsi_prev = rsi.iloc[i - 1]
        rsi_curr = rsi.iloc[i]
        pred = preds[idx]

        if (not in_trade and pd.notna(rsi_prev) and pd.notna(rsi_curr)
                and rsi_prev < 30 and rsi_curr >= 30 and pred == 1):
            shares = cash / price
            cash = 0
            in_trade = True
            trades += 1
            buy_markers.append(dict(
                x=x_axis.iloc[i],
                y=price,
                text=f"Buy(Model): {price:.2f}<br>RSI반등+모델Buy",
                type='buy'
            ))
        elif in_trade and pd.notna(rsi_curr) and rsi_curr >= 70 and pred == 2:
            cash = shares * price
            shares = 0
            in_trade = False
            trades += 1
            profit = cash - portfolio
            sell_markers.append(dict(
                x=x_axis.iloc[i],
                y=price,
                text=f"Sell(Model): {price:.2f} (RSI>=70)<br>Profit: {profit:.2f}",
                type='sell'
            ))
    final_value = cash + shares * stock_data["close"].iloc[-1]
    profit_ratio = (final_value - portfolio) / portfolio
    return {
        "name": "RSI+모델 결합 전략",
        "buy": buy_markers,
        "sell": sell_markers,
        "final_value": final_value,
        "num_trades": trades,
        "profit_ratio": profit_ratio
    }

def strategy_model_based(stock_data, x_axis, model, scaler, window_size=30):
    print("MODEL BASED===============================")

    portfolio = 100000
    cash = portfolio
    shares = 0
    in_trade = False

    buy_markers = []
    sell_markers = []
    trades = 0

    selected_features = [f.lower() for f in TrainingConfig.ALL_FEATURES]
    print(selected_features)

    valid_idx = []
    X = []
    mask = []

    for i in range(window_size, len(stock_data)):
        window = stock_data[selected_features].iloc[i - window_size:i].values
        X.append(window)
        m = (~np.isnan(window)).astype(np.float32)

        for feat in TrainingConfig.PRICE_FEATURES:
            idx = selected_features.index(feat.lower())
            m[:, idx] = 1.0
        mask.append(m)

        valid_idx.append(i)

    X = np.array(X)
    mask = ~np.isnan(X)
    price_features_lower = [f.lower() for f in TrainingConfig.PRICE_FEATURES]
    for feat in price_features_lower:
        if feat in selected_features:
            idx = selected_features.index(feat)
            mask[:, :, idx] = 1.0
    X[np.isnan(X)] = 0
    X = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

    # NOTE : 마스킹 처리
    print("X stats:", X.min(), X.max(), X.mean())
    print("Mask ratio per feature:", mask.mean(axis=(0, 1)))

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).to(device)
        masks = torch.tensor(mask, dtype=torch.float32).to(device)
        outputs = model(inputs, masks)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # NOTE : 예측 결과 분포 확인
    print("Pred value counts:", np.bincount(preds))

    for idx, i in enumerate(valid_idx):
        price = stock_data["close"].iloc[i]
        pred = preds[idx]

        if not in_trade and pred == 1:
            shares = cash / price
            cash = 0
            in_trade = True
            trades += 1
            buy_markers.append(dict(
                x=x_axis.iloc[i] if hasattr(x_axis, "iloc") else x_axis[i],
                y=price,
                text=f"Buy(Model): {price:.2f}",
                type='buy'
            ))
        elif in_trade and pred == 2:
            cash = shares * price
            shares = 0
            in_trade = False
            trades += 1
            profit = cash - portfolio
            sell_markers.append(dict(
                x=x_axis.iloc[i] if hasattr(x_axis, "iloc") else x_axis[i],
                y=price,
                text=f"Sell(Model): {price:.2f}<br>Profit: {profit:.2f}",
                type='sell'
            ))
    final_value = cash + shares * stock_data["close"].iloc[-1]
    profit_ratio = (final_value - portfolio) / portfolio
    return {
        "name": "모델 단독 매매",
        "buy": buy_markers,
        "sell": sell_markers,
        "final_value": final_value,
        "num_trades": trades,
        "profit_ratio": profit_ratio
    }


def create_comparison_chart_model(stock_data, x_axis, indicators, rule_result, model_result):
    fig = go.Figure()

    # 가격
    fig.add_trace(go.Scatter(x=x_axis, y=stock_data["close"], mode='lines', name='close', line=dict(color='black')))

    # Buy/Sell marker: 규칙(초록/빨강), 모델단독(파랑/주황)
    for marker in rule_result["buy"]:
        fig.add_trace(go.Scatter(x=[marker['x']], y=[marker['y']], mode='markers+text', name='Buy(Rule)',
                                 marker=dict(symbol='triangle-up', size=13, color='green'),
                                 text=["Buy"], hovertext=marker['text'], hoverinfo="text"))
    for marker in rule_result["sell"]:
        fig.add_trace(go.Scatter(x=[marker['x']], y=[marker['y']], mode='markers+text', name='Sell(Rule)',
                                 marker=dict(symbol='triangle-down', size=13, color='red'),
                                 text=["Sell"], hovertext=marker['text'], hoverinfo="text"))

    for marker in model_result["buy"]:
        fig.add_trace(go.Scatter(x=[marker['x']], y=[marker['y']], mode='markers+text', name='Buy(Model Only)',
                                 marker=dict(symbol='circle', size=10, color='blue'),
                                 text=["Buy(MO)"], hovertext=marker['text'], hoverinfo="text"))
    for marker in model_result["sell"]:
        fig.add_trace(go.Scatter(x=[marker['x']], y=[marker['y']], mode='markers+text', name='Sell(Model Only)',
                                 marker=dict(symbol='x', size=13, color='orange'),
                                 text=["Sell(MO)"], hovertext=marker['text'], hoverinfo="text"))

    if hasattr(x_axis, "iloc"):
        first_elem = x_axis.iloc[0]
    else:
        first_elem = x_axis[0]
    xaxis_title = "Date" if isinstance(first_elem, pd.Timestamp) else "Index"

    fig.update_layout(
        title=f"[전략 비교] <b>규칙 vs 모델 단독 매매</b><br>"
              f"규칙: {rule_result['final_value']:.0f}원({rule_result['profit_ratio']*100:.2f}%) | "
              f"모델단독: {model_result['final_value']:.0f}원({model_result['profit_ratio']*100:.2f}%)<br>"
              f"거래횟수(규칙): {rule_result['num_trades']}, 거래횟수(모델단독): {model_result['num_trades']}",
        xaxis_title=xaxis_title,
        yaxis_title="Price",
        legend=dict(x=1.05, y=1),
        hovermode="x unified",
        width=1200,
        height=650
    )
    fig.show()

if __name__ == '__main__':
    file_path = "./raw_data/stock_data_from_api_kakao.csv"
    start = time.time()
    x_axis, stock_data = load_csv(file_path)
    print("컬럼 목록:", stock_data.columns.tolist())

    # RSI만 계산
    indicators = calc_talib(stock_data, ["rsi"])
    rule_result = strategy_rsi_rebound(stock_data, x_axis, indicators)

    # 모델 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 준비 이후, model.load_state_dict(state_dict) 다음에 이 코드 추가:
    selected_features = [f.lower() for f in TrainingConfig.ALL_FEATURES]

    print("선택 피처:", selected_features)

    # MaskAwareLSTM의 인자: input_dim, hidden_dim, num_layers=2, dropout=0.3
    model = MaskAwareLSTM(input_dim=len(selected_features), hidden_dim=64, output_dim=3, num_layers=2, dropout=0.3).to(device)

    # state_dict와 scaler 경로 확인!
    state_dict = torch.load("./models/lstm_classifier.pt", map_location=device)
    print("state_dict keys:", list(state_dict.keys())[:10])

    scaler = joblib.load("./models/scaler_masked.pkl")
    model.load_state_dict(state_dict)
    print("selected_features:", selected_features)
    print("scaler.mean_[:5]:", scaler.mean_[:5])
    print("scaler.var_[:5]:", scaler.var_[:5])

    model_result = strategy_rsi_rebound_model_based(stock_data, x_axis, indicators, model, scaler, window_size=30)
    model_only_result = strategy_model_based(stock_data, x_axis, model, scaler, window_size=30)

    create_comparison_chart_model(stock_data, x_axis, indicators, rule_result, model_only_result)

    end = time.time()
    print("Total elapsed:", end - start)
