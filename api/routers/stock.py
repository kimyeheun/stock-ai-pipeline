import os

import pandas as pd
from fastapi import APIRouter
from openai import AsyncOpenAI

# 지표 계산
from api.ai.calc_indicator import add_technical_indicators
from api.ai.prompt_interpreter import prompt_bifurcation
from api.schemas import StockInitRequest, StockInitResponse, PromptRequest, PromptResponse, ActionResult
from api.store import STOCK_DATA_STORE

router = APIRouter()
client = AsyncOpenAI(base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1")


@router.post("/init", response_model=StockInitResponse)
async def stock_init(request: StockInitRequest):
    # 보조 지표 계산 후 저장.
    df = pd.DataFrame({
        "Open": request.ohlcv.open,
        "High": request.ohlcv.high,
        "Low": request.ohlcv.low,
        "Close": request.ohlcv.close,
        "Volume": request.ohlcv.volume,
    })
    df = add_technical_indicators(df)
    df.columns = [col.lower() for col in df.columns]

    # 계산된 지표를 roomId 별로 저장
    STOCK_DATA_STORE[request.roomId] = df
    print(STOCK_DATA_STORE.keys())
    print(df)
    return StockInitResponse(result="ok")


@router.post("/prompt", response_model=PromptResponse)
async def stock_prompt(request: PromptRequest):
    room_id = request.roomId

    stock_df = STOCK_DATA_STORE.get(room_id)
    results = []

    for user_prompt in request.prompts:
        user_id = user_prompt.userId
        prompt = user_prompt.prompt

        actions = await prompt_bifurcation(prompt, stock_df, client)
        print("================actions===============")
        print(actions)
        results.append(ActionResult(userId=user_id, action=actions))

    return PromptResponse(results=results)