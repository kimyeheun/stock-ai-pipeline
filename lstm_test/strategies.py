import pandas as pd
import torch

from BaseStrategy import BaseStrategy
from Config import TestConfig


# NOTE: 상급자
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


# NOTE: 중급자
class IntermediateStrategy(BaseStrategy):
    def run(self, stock_data, model, scaler, window_size=30, indicators=None, **kwargs):
        self.reset()

        if indicators is None:
            indicators = []
        valid_idx = []
        X = []
        mask = []
        x_axis = stock_data.index

        for i in range(window_size, len(stock_data)):
            window = stock_data[TestConfig.ALL_FEATURES].iloc[i - window_size:i].values
            X.append(window)
            m = (~np.isnan(window)).astype(np.float32)
            for feat in TestConfig.PRICE_FEATURES:
                idx = TestConfig.ALL_FEATURES.index(feat)
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

        rsi = indicators.get("RSI")
        for idx, i in enumerate(valid_idx[1:], 1):
            price = stock_data["Close"].iloc[i]
            rsi_prev = rsi.iloc[i - 1]
            rsi_curr = rsi.iloc[i]
            pred = preds[idx]
            if (not self.in_trade and pd.notna(rsi_prev) and pd.notna(rsi_curr)
                    and rsi_prev < 30 and rsi_curr >= 30 and pred == 1):
                self.shares = self.cash / price
                self.cash = 0
                self.in_trade = True
                self.trades += 1
                self.buy_markers.append(dict(
                    x=x_axis.iloc[i] if hasattr(x_axis, "iloc") else x_axis[i],
                    y=price,
                    text=f"Buy(Model): {price:.2f}<br>RSI반등+모델Buy",
                    type='buy'
                ))
            elif self.in_trade and pd.notna(rsi_curr) and rsi_curr >= 70 and pred == 2:
                self.cash = self.shares * price
                self.shares = 0
                self.in_trade = False
                self.trades += 1
                profit = self.cash - self.initial_cash
                self.sell_markers.append(dict(
                    x=x_axis.iloc[i] if hasattr(x_axis, "iloc") else x_axis[i],
                    y=price,
                    text=f"Sell(Model): {price:.2f} (RSI>=70)<br>Profit: {profit:.2f}",
                    type='sell'
                ))
        return self.result(stock_data)


import numpy as np





# NOTE: 초급자
class LowerStrategy(BaseStrategy):
    @staticmethod
    def get_feature_mask(mask, all_features, indicators):
        use_features = TestConfig.PRICE_FEATURES + indicators
        for idx, feat in enumerate(all_features):
            if feat not in use_features:
                mask[:, idx] = 0.0
        return mask

    def run(self, stock_data, model, scaler, window_size=30, indicators=None, **kwargs):
        self.reset()

        indicators = [] if indicators is None else indicators
        valid_idx = []
        X, mask = [], []
        x_axis = stock_data.index

        window_mask = LowerStrategy.get_feature_mask(
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
        print("preds", preds)

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
