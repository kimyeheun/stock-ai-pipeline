import pandas as pd
import plotly.graph_objects as go


def create_comparison_chart_model(stock_data, x_axis, rule_result, model_result, hybrid_result=None):
    fig = go.Figure()

    # 가격
    fig.add_trace(go.Scatter(x=x_axis, y=stock_data["Close"], mode='lines', name='close', line=dict(color='black')))

    # Buy/Sell marker: 규칙(초록/빨강), 모델단독(파랑/주황)
    for marker in rule_result["buy"]:
        fig.add_trace(go.Scatter(x=[marker['x']], y=[marker['y']], mode='markers+text', name='Buy(Rule)',
                                 marker=dict(symbol='triangle-up', size=13, color='green'),
                                 text=["Buy"], hovertext=marker['text'], hoverinfo="text"))
    for marker in rule_result["sell"]:
        fig.add_trace(go.Scatter(x=[marker['x']], y=[marker['y']], mode='markers+text', name='Sell(Rule)',
                                 marker=dict(symbol='triangle-down', size=13, color='red'),
                                 text=["Sell"], hovertext=marker['text'], hoverinfo="text"))

    # for marker in model_result["buy"]:
    #     fig.add_trace(go.Scatter(x=[marker['x']], y=[marker['y']], mode='markers+text', name='Buy(Model Only)',
    #                              marker=dict(symbol='circle', size=10, color='blue'),
    #                              text=["Buy(MO)"], hovertext=marker['text'], hoverinfo="text"))
    # for marker in model_result["sell"]:
    #     fig.add_trace(go.Scatter(x=[marker['x']], y=[marker['y']], mode='markers+text', name='Sell(Model Only)',
    #                              marker=dict(symbol='x', size=13, color='orange'),
    #                              text=["Sell(MO)"], hovertext=marker['text'], hoverinfo="text"))

    if hybrid_result is not None:
        for marker in hybrid_result["buy"]:
            fig.add_trace(go.Scatter(x=[marker['x']], y=[marker['y']], mode='markers+text', name='Buy(Hybrid)',
                                     marker=dict(symbol='star', size=12, color='pink'),
                                     text=["Buy(MO)"], hovertext=marker['text'], hoverinfo="text"))
        for marker in hybrid_result["sell"]:
            fig.add_trace(go.Scatter(x=[marker['x']], y=[marker['y']], mode='markers+text', name='Sell(Hybrid)',
                                     marker=dict(symbol='star', size=12, color='black'),
                                     text=["Sell(MO)"], hovertext=marker['text'], hoverinfo="text"))

    if hasattr(x_axis, "iloc"):
        first_elem = x_axis.iloc[0]
    else:
        first_elem = x_axis[0]
    xaxis_title = "Date" if isinstance(first_elem, pd.Timestamp) else "Index"

    # title 생성 (hybrid 결과 포함)
    if hybrid_result is not None:
        title_text = (f"[전략 비교] <b>규칙 vs 모델 단독 vs 하이브리드 매매</b><br>"
                     f"규칙: {rule_result['final_value']:.0f}원({rule_result['profit_ratio']*100:.2f}%) | "
                     f"모델단독: {model_result['final_value']:.0f}원({model_result['profit_ratio']*100:.2f}%) | "
                     f"하이브리드: {hybrid_result['final_value']:.0f}원({hybrid_result['profit_ratio']*100:.2f}%)<br>"
                     f"거래횟수 - 규칙: {rule_result['num_trades']}, 모델단독: {model_result['num_trades']}, 하이브리드: {hybrid_result['num_trades']}")
    else:
        title_text = (f"[전략 비교] <b>규칙 vs 모델 단독 매매</b><br>"
                     f"규칙: {rule_result['final_value']:.0f}원({rule_result['profit_ratio']*100:.2f}%) | "
                     f"모델단독: {model_result['final_value']:.0f}원({model_result['profit_ratio']*100:.2f}%)<br>"
                     f"거래횟수(규칙): {rule_result['num_trades']}, 거래횟수(모델단독): {model_result['num_trades']}")

    fig.update_layout(
        title=title_text,
        xaxis_title=xaxis_title,
        yaxis_title="Price",
        legend=dict(x=1.05, y=1),
        hovermode="x unified",
        width=1200,
        height=650
    )
    fig.show()
