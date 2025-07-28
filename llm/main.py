import asyncio
import getpass
import os

import yfinance as yf
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI

from prompt import generate_dsl


async def main():
    # 1. 주식 데이터 다운로드
    df = yf.download("AAPL", start="2023-01-01", end="2023-12-31", auto_adjust=False)
    df.columns = df.columns.get_level_values(0)
    df.rename(columns=str.lower, inplace=True)

    # 2. 사용자 입력 전략
    natural_text = "RSI가 30 밑에서 반등. MACD 우상향이면 매수"

    # 3. LLM → DSL 파싱
    dsl = await generate_dsl(client, natural_text)
    print("=====DSL=====")
    print(dsl)
    print()
    #
    #
    # # 4. DSL → Python 전략 코드 생성
    # code = generate_dsl(dsl.conditions, df_var='df')
    #
    # print("🔧 생성된 전략 코드:\n")
    # print(code)
    #
    # # 5. 실행 환경 구성 및 코드 실행
    # exec_env = {'talib': talib, 'df': df, 'pd': pd}
    # exec(code, exec_env)
    #
    # # 6. 결과 시각 확인
    # df['Buy'] = exec_env['final_buy_signal']
    # df['Sell'] = exec_env['final_sell_signal']
    #
    # print("\n📈 시그널 출력 (최근 10개):")
    # print(df[['close', 'Buy', 'Sell']].tail(10))


if __name__ == "__main__":
    # 🔑 OpenAI Key 설정
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("GMS KEY를 입력하세요: ")

    client = AsyncOpenAI(base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1")

    asyncio.run(main())
