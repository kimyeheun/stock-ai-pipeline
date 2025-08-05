# NOTE: 프롬프트 분기점
import re
from typing import List, Any, Coroutine

from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from api.ai.prompt_interpreter import professional_level


# 상 중 하 분류 후 각 코드로 분기
async def classify_difficulty(prompt_text: str) -> int:
    llm = ChatOllama(
        model="llama3.2",
        mirostat=1,
    )
    template = PromptTemplate(
        input_variables=["prompt_text"],
        template="""아래 명령을 보고 0, 1, 2로 분류해라. 반드시 숫자만 답하라.
- 0: 매매 기준이 없음 + 감각적 표현, 지표 없음, 매매 책임을 타인에게 위임하거나, 전문가라도 스스로 매매 판단을 포기함
- 1: 매매 기준이 있음 + 지표 언급, 단일 기준
- 2: 매매 기준이 있음 + 복합 기준, 여러 지표 조합
주의 해야 할 점은, 매매 기준이 없으면 무조건 0이라는거야. 
예시:
내려갈 것 같으면 사지 마. 오를 때 들어가줘. : 0
나는 MACD, RSI 같은거 잘 모르겠으니까 알아서 사줘 : 0
MACD 골든크로스 예상되니 선매수. 데드크로스 나오면 바로 매도. : 1
MACD 골든크로스 발생 직후에 진입. 볼린저밴드 상단 닿으면 일부 익절, 반전 시 전량 매도. : 2

{prompt_text} : """)

    chain = template | llm
    result = chain.invoke({"prompt_text": prompt_text}).content
    print(result)

    m = re.search(r'\b([0-2])\b', result)
    if m:
        return int(m.group(1))
    m2 = re.search(r'([0-2])', result)
    if m2:
        return int(m2.group(1))
    # raise ValueError(f"정상적인 0, 1, 2 답변이 아닙니다: {result!r}")
    return 0


# 프롬프트 분기점
async def prompt_bifurcation(prompt_text: str, stock_df, client) -> List:
    # 1. 프롬프트 난이도 분류
    difficulty = await classify_difficulty(prompt_text)
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

