# NOTE: 각 단계별 로직 모음

import os
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from langchain.chains import SequentialChain

from llm.dsl_interpreter_2 import dsl_to_code
from llm.prompt import generate_dsl
from lstm_test.Model import MaskAwareLSTM
from lstm_test.strategies import LowerStrategy


# NOTE:프롬프트 분기점
async def prompt_bifurcation(difficulty : int, prompt_text:str, stock_df, client) -> List:
    if difficulty == 2:
        action = await upper_level(prompt_text, stock_df, client)
    elif difficulty == 1:
        action = await intermediate_level(prompt_text, stock_df)
    elif difficulty == 0:
        action = await lower_level(prompt_text, stock_df)
    else:
        raise NotImplementedError("난이도가 잘못 출력되었습니다. 복구 로직을 실행하세요.")
    return action[-60:]

# NOTE: 상
# 메인 액션 추론 함수 (상 난이도만, stock_df는 OHLCV+지표 DataFrame)
async def upper_level(prompt_text:str, stock_df:pd.DataFrame, openai_client) -> List:
    # 1. LLM → DSL 파싱
    dsl = await generate_dsl(openai_client, prompt_text)

    # 2. DSL → 코드 변환
    code = dsl_to_code(dsl, df_var="df")
    print("============code=============")
    print(code)
    # 3. 코드 실행 환경 준비 및 신호 추론
    local_env = {
        "df": stock_df,
        "np": np,
        "pd": pd,
        "talib": __import__("talib")  # TA-Lib 파이썬 래퍼
    }

    exec(code, local_env)

    # 4. Buy/Sell 시그널 불리언 시리즈를 int로 변환 (0: 유지, 1: 매수, 2: 매도)
    buy_signal = local_env.get("final_buy_signal", pd.Series([False]*len(stock_df)))
    sell_signal = local_env.get("final_sell_signal", pd.Series([False]*len(stock_df)))

    # TODO: 예외 처리 어떻게 할까...
    # [0,1,2] 리스트 생성 (우선 순위: 매수>매도>유지, 혹은 동시=매수)
    action = []
    for b, s in zip(buy_signal, sell_signal):
        if b and not s:
            action.append(1)  # 매수
        elif not b and s:
            action.append(2)  # 매도
        elif b and s:
            action.append(1)  # 동시시 매수 우선(예시)
        else:
            action.append(0)  # 유지
    return action

# NOTE: 중
async def intermediate_level(prompt_text:str, stock_df) -> List:
    return []

# NOTE: 하
async def lower_level(prompt_text:str, stock_df) -> List:
    # TODO : indicator 프롬프트에서 뽑아내기
    use_indicators = ["RSI", "MACD", "MACD_SIGNAL", "BB_UPPER", "BB_LOWER", "MOM", "CCI"]
    current_dir = os.path.dirname(os.path.abspath(__file__))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskAwareLSTM(input_dim=12, hidden_dim=64, output_dim=3, num_layers=2, dropout=0.3).to(device)

    model_path = os.path.join(current_dir, '..', 'model', 'lstm_classifier.pt')
    scaler_path = os.path.join(current_dir, '..', 'model', 'scaler.pkl')

    # state_dict와 scaler 경로 확인!
    state_dict = torch.load(model_path , map_location=device)
    scaler = joblib.load(scaler_path)
    model.load_state_dict(state_dict)

    # 모델 추론
    lower_level = LowerStrategy()
    action = lower_level.run_for_api(stock_df, model, scaler,
                                       window_size=30,
                                       indicators=use_indicators)

    print(action)
    return action