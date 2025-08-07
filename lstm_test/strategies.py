import numpy as np
import pandas as pd
import torch

from lstm_test.BaseStrategy import BaseStrategy
from lstm_test.Config import TestConfig


# NOTE: 상급자 (일단은 RSI만 적용됨. 실제로는 DSL -> Code)
class UpperStrategy(BaseStrategy):
    def run(self, stock_data, **kwargs):
        self.reset()

        x_axis = stock_data.index
        rsi = stock_data.get("RSI")

        for i in range(1, len(stock_data)):
            price = stock_data["Close"].iloc[i]
            rsi_prev = rsi.iloc[i-1]
            rsi_curr = rsi.iloc[i]

            if (not self.in_trade and pd.notna(rsi_prev) and pd.notna(rsi_curr) and rsi_prev < 30 and rsi_curr >= 30):
                self.shares = self.cash / price
                self.cash = 0
                self.in_trade = True
                self.trades += 1
                self.buy_markers.append(dict(
                    x=x_axis[i],
                    y=price,
                    text=f"Buy: {price:.2f} (RSI 반등)",
                    type='buy'
                ))
            elif self.in_trade and pd.notna(rsi_curr) and rsi_curr >= 70:
                self.cash = self.shares * price
                self.shares = 0
                self.in_trade = False
                self.trades += 1
                profit = self.cash - self.initial_cash
                self.sell_markers.append(dict(
                    x=x_axis[i],
                    y=price,
                    text=f"Sell: {price:.2f} (RSI>=70)<br>Profit: {profit:.2f}",
                    type='sell'
                ))
        return self.result(stock_data)

def get_feature_mask(mask, all_features, indicators):
    use_features = TestConfig.PRICE_FEATURES + indicators
    for idx, feat in enumerate(all_features):
        if feat not in use_features:
            mask[:, idx] = 0.0
    return mask

# NOTE: 중급자
class IntermediateStrategy(BaseStrategy):
    # def run(self, stock_data, model, scaler, window_size=30, indicators=None, **kwargs):
    #     self.reset()
    #
    #     indicators = [] if indicators is None else indicators
    #     valid_idx = []
    #     X, mask = [], []
    #     x_axis = stock_data.index
    #
    #     window_mask = get_feature_mask(
    #         np.ones((window_size, len(TestConfig.ALL_FEATURES)), dtype=np.float32),
    #         TestConfig.ALL_FEATURES, indicators)
    #
    #     for i in range(window_size, len(stock_data)):
    #         window = stock_data[TestConfig.ALL_FEATURES].iloc[i - window_size:i].values
    #         X.append(window)
    #         mask.append(window_mask)
    #         valid_idx.append(i)
    #
    #     mask = np.array(mask)
    #
    #     X = np.array(X)
    #     X[np.isnan(X)] = 0
    #     X = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    #
    #     model.eval()
    #     device = next(model.parameters()).device
    #     with torch.no_grad():
    #         inputs = torch.tensor(X, dtype=torch.float32).to(device)
    #         masks = torch.tensor(mask, dtype=torch.float32).to(device)
    #         outputs = model(inputs, masks)
    #         preds = torch.argmax(outputs, dim=1).cpu().numpy()
    #
    #     rsi = stock_data.get("RSI")
    #     for idx, i in enumerate(valid_idx[1:], 1):
    #         price = stock_data["Close"].iloc[i]
    #         rsi_prev = rsi.iloc[i - 1]
    #         rsi_curr = rsi.iloc[i]
    #         pred = preds[idx]
    #         if (not self.in_trade and pd.notna(rsi_prev) and pd.notna(rsi_curr)
    #                 and rsi_prev < 30 and rsi_curr >= 30 and pred == 1):
    #             self.shares = self.cash / price
    #             self.cash = 0
    #             self.in_trade = True
    #             self.trades += 1
    #             self.buy_markers.append(dict(
    #                 x=x_axis.iloc[i] if hasattr(x_axis, "iloc") else x_axis[i],
    #                 y=price,
    #                 text=f"Buy(Model): {price:.2f}<br>RSI반등+모델Buy",
    #                 type='buy'
    #             ))
    #         elif self.in_trade and pd.notna(rsi_curr) and rsi_curr >= 70 and pred == 2:
    #             self.cash = self.shares * price
    #             self.shares = 0
    #             self.in_trade = False
    #             self.trades += 1
    #             profit = self.cash - self.initial_cash
    #             self.sell_markers.append(dict(
    #                 x=x_axis.iloc[i] if hasattr(x_axis, "iloc") else x_axis[i],
    #                 y=price,
    #                 text=f"Sell(Model): {price:.2f} (RSI>=70)<br>Profit: {profit:.2f}",
    #                 type='sell'
    #             ))
    #     return self.result(stock_data)
    def run(self, stock_data, model, scaler, window_size=30, indicators=None, **kwargs):
        self.reset()

        indicators = [] if indicators is None else indicators
        valid_idx = []
        X, mask = [], []
        x_axis = stock_data.index

        window_mask = get_feature_mask(
            np.ones((window_size, len(TestConfig.ALL_FEATURES)), dtype=np.float32),
            TestConfig.ALL_FEATURES, indicators)

        for i in range(window_size, len(stock_data)):
            window = stock_data[TestConfig.ALL_FEATURES].iloc[i - window_size:i].values
            X.append(window)
            mask.append(window_mask)
            valid_idx.append(i)

        mask = np.array(mask)

        X = np.array(X)
        X[np.isnan(X)] = 0
        X = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            masks = torch.tensor(mask, dtype=torch.float32).to(device)
            outputs = model(inputs, masks)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        rsi = stock_data.get("RSI")
        i = 1
        while i < len(stock_data):
            # [매수 후보군] 규칙 신호 발생 (RSI < 30 → >= 30) 구간 탐색
            if (not self.in_trade and pd.notna(rsi.iloc[i - 1]) and pd.notna(rsi.iloc[i]) and
                    rsi.iloc[i - 1] < 30 and rsi.iloc[i] >= 30):
                # 후보구간: 규칙 신호 발생시점~최대 N일 이내(혹은 조건 만족 구간)
                candidate_idx = []
                lookahead = 7  # 규칙 신호 후 며칠간 모델 타이밍을 탐색할지
                for j in range(i, min(i + lookahead, len(stock_data))):
                    # valid_idx에 해당 j가 있는지 확인
                    if j in valid_idx:
                        idx_in_X = valid_idx.index(j)
                        if preds[idx_in_X] == 1:
                            candidate_idx.append(j)
                # 후보군 중 가장 첫 Buy 신호(혹은 확률 최대 등)에서 진입
                if candidate_idx:
                    buy_i = candidate_idx[0]
                    price = stock_data["Close"].iloc[buy_i]
                    self.shares = self.cash / price
                    self.cash = 0
                    self.in_trade = True
                    self.trades += 1
                    self.buy_markers.append(dict(
                        x=x_axis.iloc[buy_i] if hasattr(x_axis, "iloc") else x_axis[buy_i],
                        y=price,
                        text=f"Buy: {price:.2f} (RSI반등+모델)",
                        type='buy'
                    ))
                    i = buy_i + 1  # 진입 후 다음 탐색으로 점프
                    continue
            # [매도 후보군] 보유중일 때, 규칙(RSI >= 70) 충족 시점
            if self.in_trade and pd.notna(rsi.iloc[i]) and rsi.iloc[i] >= 70:
                candidate_idx = []
                lookahead = 5  # 규칙 신호 후 며칠간 모델 타이밍을 탐색할지
                for j in range(i, min(i + lookahead, len(stock_data))):
                    if j in valid_idx:
                        idx_in_X = valid_idx.index(j)
                        if preds[idx_in_X] == 2:
                            candidate_idx.append(j)
                if candidate_idx:
                    sell_i = candidate_idx[0]
                    price = stock_data["Close"].iloc[sell_i]
                    self.cash = self.shares * price
                    self.shares = 0
                    self.in_trade = False
                    self.trades += 1
                    profit = self.cash - self.initial_cash
                    self.sell_markers.append(dict(
                        x=x_axis.iloc[sell_i] if hasattr(x_axis, "iloc") else x_axis[sell_i],
                        y=price,
                        text=f"Sell: {price:.2f} (RSI>=70+모델)<br>Profit: {profit:.2f}",
                        type='sell'
                    ))
                    i = sell_i + 1  # 매도 후 다음 탐색으로 점프
                    continue
            i += 1
        return self.result(stock_data)


# NOTE: 초급자
class LowerStrategy(BaseStrategy):

    def run(self, stock_data, model, scaler, window_size=30, indicators=None, **kwargs):
        self.reset()

        indicators = [] if indicators is None else indicators
        valid_idx = []
        X, mask = [], []
        x_axis = stock_data.index

        window_mask = get_feature_mask(
            np.ones((window_size, len(TestConfig.ALL_FEATURES)), dtype=np.float32),
            TestConfig.ALL_FEATURES, indicators)

        for i in range(window_size, len(stock_data)):
            window = stock_data[TestConfig.ALL_FEATURES].iloc[i - window_size:i].values
            X.append(window)
            mask.append(window_mask)
            valid_idx.append(i)

        mask = np.array(mask)

        X = np.array(X)
        X[np.isnan(X)] = 0
        X = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            masks = torch.tensor(mask, dtype=torch.float32).to(device)
            outputs = model(inputs, masks)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        for idx, i in enumerate(valid_idx):
            price = stock_data["Close"].iloc[i]
            pred = preds[idx]
            if not self.in_trade and pred == 1:
                self.shares = self.cash / price
                self.cash = 0
                self.in_trade = True
                self.trades += 1
                self.buy_markers.append(dict(
                    x=x_axis.iloc[i] if hasattr(x_axis, "iloc") else x_axis[i],
                    y=price,
                    text=f"Buy(Model): {price:.2f}",
                    type='buy'
                ))
            elif self.in_trade and pred == 2:
                self.cash = self.shares * price
                self.shares = 0
                self.in_trade = False
                self.trades += 1
                profit = self.cash - self.initial_cash
                self.sell_markers.append(dict(
                    x=x_axis.iloc[i] if hasattr(x_axis, "iloc") else x_axis[i],
                    y=price,
                    text=f"Sell(Model): {price:.2f}<br>Profit: {profit:.2f}",
                    type='sell'
                ))
        return self.result(stock_data)

    def run_for_api(self, stock_data, model, scaler, window_size=30, indicators=None, **kwargs):
        self.reset()

        indicators = [] if indicators is None else indicators
        valid_idx = []
        X, mask = [], []

        window_mask = get_feature_mask(
            np.ones((window_size, len(TestConfig.ALL_FEATURES)), dtype=np.float32),
            TestConfig.ALL_FEATURES, indicators)

        for i in range(window_size, len(stock_data)):
            window = stock_data[TestConfig.ALL_FEATURES].iloc[i - window_size:i].values
            X.append(window)
            mask.append(window_mask)
            valid_idx.append(i)

        mask = np.array(mask)

        X = np.array(X)
        X[np.isnan(X)] = 0
        X = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            masks = torch.tensor(mask, dtype=torch.float32).to(device)
            outputs = model(inputs, masks)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        # 수정된 부분 시작
        # 0(유지), 1(매수), 2(매도)로 구성된 리스트를 생성합니다.
        # 초기값은 모두 유지(0)로 설정합니다.
        predictions_list = [0] * len(stock_data)

        for idx, i in enumerate(valid_idx):
            pred = preds[idx]

            # 모델 예측에 따라 1(매수) 또는 2(매도)를 할당합니다.
            # 여기서는 매수(1)와 매도(2)만 고려하며, 우선순위를 적용하지 않고 단순 할당합니다.
            if pred == 1:
                predictions_list[i] = 1  # 매수
            elif pred == 2:
                predictions_list[i] = 2  # 매도

        return predictions_list