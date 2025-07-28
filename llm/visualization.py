import pandas as pd
import talib
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots


df = yf.download("AAPL", start="2023-01-01", end="2023-12-31", auto_adjust=False)
df.rename(columns=str.lower, inplace=True)
df.columns = [col[0].lower() for col in df.columns]

print(df.columns)
print(df['close'].shape)
print(df['close'].isnull().sum())

code = """
sma = talib.SMA(df['close'], timeperiod=20)
sma60 = talib.SMA(df['close'], timeperiod=60)
rsi = talib.RSI(df['close'], timeperiod=14)
entry_signal_0 = (((sma.shift(1) <= sma60.shift(1)) & (sma > sma60))&((rsi <= 60) & (rsi > rsi.shift(1))))
macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
exit_signal_0 = ((rsi >= 70)|(1 & (macd < macd.shift(1))))
final_buy_signal = entry_signal_0
final_sell_signal = exit_signal_0
"""


exec_env = {'talib': talib, 'df': df, 'pd': pd}
exec(code, exec_env)

# 6. 결과 시각 확인
df['Buy'] = exec_env['final_buy_signal']
df['Sell'] = exec_env['final_sell_signal']

print("\n📈 시그널 출력 (최근 10개):")
print(df[['close', 'Buy', 'Sell']].tail(500))
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', None,
                       'display.max_colwidth', None):
    print(df[['close', 'Buy', 'Sell']])


# NOTE: ========================== 시각화 ========================
import pandas as pd
import talib
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 데이터 로드 및 전처리
df = yf.download("AAPL", start="2023-01-01", end="2023-12-31", auto_adjust=False)
df.rename(columns=str.lower, inplace=True)
df.columns = [col[0].lower() for col in df.columns]

print(f"데이터 로드 완료: {df.shape[0]}개 행")

# 매매 신호 생성
code = """
sma = talib.SMA(df['close'], timeperiod=20)
sma60 = talib.SMA(df['close'], timeperiod=60)
rsi = talib.RSI(df['close'], timeperiod=14)
entry_signal_0 = (((sma.shift(1) <= sma60.shift(1)) & (sma > sma60))&((rsi <= 60) & (rsi > rsi.shift(1))))
macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
exit_signal_0 = ((rsi >= 70)|(1 & (macd < macd.shift(1))))
final_buy_signal = entry_signal_0
final_sell_signal = exit_signal_0
"""

exec_env = {'talib': talib, 'df': df, 'pd': pd}
exec(code, exec_env)

# 신호 추가
df['Buy'] = exec_env['final_buy_signal']
df['Sell'] = exec_env['final_sell_signal']
df['SMA20'] = exec_env['sma']
df['SMA60'] = exec_env['sma60']
df['RSI'] = exec_env['rsi']
df['MACD'] = exec_env['macd']
df['MACD_Signal'] = exec_env['macd_signal']

# Buy/Sell 신호 포인트 추출
buy_signals = df[df['Buy'] == True]
sell_signals = df[df['Sell'] == True]

print(f"📈 Buy 신호: {len(buy_signals)}개")
print(f"📉 Sell 신호: {len(sell_signals)}개")

# 서브플롯 생성 (가격 차트 + RSI + MACD)
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=('AAPL 주가 및 매매신호', 'RSI (14)', 'MACD'),
    row_heights=[0.6, 0.2, 0.2]
)

# 1. 주가 차트
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['close'],
        mode='lines',
        name='AAPL Close',
        line=dict(color='black', width=1)
    ),
    row=1, col=1
)

# 2. 이동평균선
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['SMA20'],
        mode='lines',
        name='SMA20',
        line=dict(color='blue', width=1, dash='dot')
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['SMA60'],
        mode='lines',
        name='SMA60',
        line=dict(color='orange', width=1, dash='dash')
    ),
    row=1, col=1
)

# 3. Buy 신호 표시
if len(buy_signals) > 0:
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals['close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
            )
        ),
        row=1, col=1
    )

# 4. Sell 신호 표시
if len(sell_signals) > 0:
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals['close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
            )
        ),
        row=1, col=1
    )

# 5. RSI 차트
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=1)
    ),
    row=2, col=1
)

# RSI 기준선 (30, 70)
fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
fig.add_hline(y=60, line_dash="dot", line_color="blue", opacity=0.3, row=2, col=1)

# 6. MACD 차트
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='blue', width=1)
    ),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['MACD_Signal'],
        mode='lines',
        name='MACD Signal',
        line=dict(color='red', width=1)
    ),
    row=3, col=1
)

# MACD 0선
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

# 레이아웃 설정
fig.update_layout(
    title={
        'text': 'AAPL 매매전략 백테스팅 결과 (2023년)',
        'x': 0.5,
        'font': {'size': 18}
    },
    height=800,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Y축 라벨 설정
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
fig.update_yaxes(title_text="MACD", row=3, col=1)
fig.update_xaxes(title_text="Date", row=3, col=1)

# 차트 표시
fig.show()

# 매매 신호 상세 분석
print("\n" + "=" * 60)
print("📊 매매 신호 상세 분석")
print("=" * 60)

if len(buy_signals) > 0:
    print(f"\n🟢 Buy 신호 발생 일자:")
    for date, row in buy_signals.iterrows():
        print(f"  📅 {date.strftime('%Y-%m-%d')}: ${row['close']:.2f} (RSI: {row['RSI']:.1f})")

if len(sell_signals) > 0:
    print(f"\n🔴 Sell 신호 발생 일자:")
    for date, row in sell_signals.iterrows():
        print(f"  📅 {date.strftime('%Y-%m-%d')}: ${row['close']:.2f} (RSI: {row['RSI']:.1f})")


# 간단한 백테스팅 성과 계산
def simple_backtest(df):
    position = 0  # 0: 현금, 1: 보유
    cash = 10000  # 초기 자금
    shares = 0
    transactions = []

    for date, row in df.iterrows():
        if row['Buy'] and position == 0:  # 매수
            shares = cash / row['close']
            cash = 0
            position = 1
            transactions.append(('BUY', date, row['close'], shares))

        elif row['Sell'] and position == 1:  # 매도
            cash = shares * row['close']
            shares = 0
            position = 0
            transactions.append(('SELL', date, row['close'], cash))

    # 마지막에 포지션이 있으면 청산
    if position == 1:
        cash = shares * df['close'].iloc[-1]
        transactions.append(('SELL', df.index[-1], df['close'].iloc[-1], cash))

    return cash, transactions


final_value, transactions = simple_backtest(df)
initial_value = 10000
total_return = (final_value - initial_value) / initial_value * 100

print(f"\n" + "=" * 60)
print("💰 간단 백테스팅 결과")
print("=" * 60)
print(f"초기 자금: ${initial_value:,.2f}")
print(f"최종 자금: ${final_value:,.2f}")
print(f"총 수익률: {total_return:.2f}%")
print(f"거래 횟수: {len(transactions)}회")

# Buy & Hold 비교
buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
print(f"Buy & Hold 수익률: {buy_hold_return:.2f}%")
print(f"전략 대비 B&H: {buy_hold_return - total_return:.2f}%p {'우수' if buy_hold_return > total_return else '열세'}")

print(f"\n거래 내역:")
for action, date, price, amount in transactions:
    if action == 'BUY':
        print(f"  📈 {date.strftime('%Y-%m-%d')}: {action} at ${price:.2f} ({amount:.2f} shares)")
    else:
        print(f"  📉 {date.strftime('%Y-%m-%d')}: {action} at ${price:.2f} (${amount:.2f} cash)")

# 추가 차트 - 수익률 곡선 (옵션)
print(f"\n💡 차트 해석:")
print("- 🟢 초록 삼각형: Buy 신호 (SMA20이 SMA60 돌파 + RSI≤60 상승)")
print("- 🔴 빨간 삼각형: Sell 신호 (RSI≥70 또는 MACD 하락)")
print("- 파란 점선: SMA20, 주황 점선: SMA60")
print("- RSI 차트에서 60선(파란 점선) 터치 후 상승 시 매수 조건")