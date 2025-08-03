import numpy as np
import pandas as pd

from llm.dsl_interpreter_2 import dsl_to_code
from llm.prompt import generate_dsl


# 난이도 분류용 함수
async def classify_difficulty(prompt_text: str, client) -> str:
    # TODO: 실제 프롬프트 로직으로 변경!!
    # system_msg = "다음 프롬프트를 난이도별로 '상', '중', '하' 중 하나로만 대답하세요."
    # response = await client.chat.completions.create(
    #     model="gpt-4.1",
    #     messages=[
    #         {"role": "system", "content": system_msg},
    #         {"role": "user", "content": prompt_text}
    #     ]
    # )
    # return response.choices[0].message.content.strip()
    return "상"

# 프롬프트 분기점
async def prompt_bifurcation(prompt_text: str, stock_df, client):
    # 1. 프롬프트 난이도 분류
    difficulty = await classify_difficulty(prompt_text, client)
    action = []
    # 2. 난이도가 "상"이 아니면, 처리하지 않음 (확장 가능)
    if difficulty == "상":
        action = await professional_level(prompt_text, stock_df, client)
    elif difficulty == "중 ":
        action = [] # TODO : 중급 난이도로 변경
    elif difficulty == "하":
        action = [] # TODO : 하급 난이도로 변경
    else:
        raise NotImplementedError("난이도가 잘못 출력되었습니다. 복구 로직을 실행하세요.")
    return action

# 메인 액션 추론 함수 (상 난이도만, stock_df는 OHLCV+지표 DataFrame)
async def professional_level(prompt_text, stock_df, openai_client):
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
    print("============local_env=============")
    print(local_env)

    exec(code, local_env)

    # 4. Buy/Sell 시그널 불리언 시리즈를 int로 변환 (0: 유지, 1: 매수, 2: 매도)
    buy_signal = local_env.get("final_buy_signal", pd.Series([False]*len(stock_df)))
    sell_signal = local_env.get("final_sell_signal", pd.Series([False]*len(stock_df)))

    # [0,1,2] 리스트 생성 (우선순위: 매수>매도>유지, 혹은 동시=매수)
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
