# NOTE: 프롬프트 분기점
import json
import logging
import re
from json import JSONDecodeError
from typing import Any, Coroutine

from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_ollama import ChatOllama


# NOTE: 상 중 하 분류 후 각 코드로 분기
'''
llm = ChatOpenAI()
examples =[ {"question": "", "answer": "",}, ]
example_prompt = PromptTemplate(
    input_variables=["question", "answer"], 
    template="Question: {question}\n{answer}",
)
prompt = FewShotPromptTemplate(
    examples examples, 
    example_prompt=example_prompt,
    suffix="Question: {question}",
    input_variables=["question"],
)

question = "7x92?"
chain = prompt | llm | StrOutputParser() 
response = chain.invoke({"question": question}) 
print(response)
'''

async def classify_difficulty(prompt_text: str) -> int:
    llm = ChatOllama(
        model="llama3.2",
        mirostat=1,
        temperature=0,
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
    return 0  # raise ValueError(f"정상적인 0, 1, 2 답변이 아닙니다: {result!r}")


# NOTE: 투자 성향 분석하기
async def get_propensity(prompt_text: str) -> tuple[Any, Any]:
    llm = ChatOllama(
        model="llama3.2",  # "martain7r/finance-llama-8b:q4_k_m",
        base_url="http://localhost:11434",
        mirostat=1,
        temperature=0,
        format="json",
    )

    template = PromptTemplate(
        input_variables=["prompt_text"],
        template="""다음 지침에 따라 문장으로부터 매수 매도 비율을 정하여라.
지침서:
1. 문장을 읽고 이해할 것. 매도 == sell, 매수 == buy
2. 매매 비율에 대한 정보가 나와 있다면, 그를 퍼센트로 변환하여 반환할 것.
3. 관련 정보가 없다면, 사람의 성격을 분석하고 그를 수치를 정할 것.
4. 수치는 퍼센트(%)로 나타낼 것!

Tip:
1.00은 내가 가진 모든 재산으로 주식을 사겠다는 의미이고, 0.00은 한 주도 사지 않겠다는 의미이다. 

예시:
"question": "전부 다 매수 할래", "answer": {{"buy": 1.00, "sell": 0.00}}
"question": "전액 매도하고, 조금만 매수 할래", "answer": {{"buy": 0.20, "sell": 1.00}}
"question": "40퍼센트 사고, 많이 팔래", "answer": {{"buy": 0.40, "sell": 0.80}}
"question": "30%씩 사고 5%씩 손절", "answer": {{"buy": 0.30, "sell": 0.05}}
"question": "도전적으로 사고 팔아보자", "answer": {{"buy": 1.00, "sell": 1.00}}
"question": "오늘은 관망", "answer": {{"buy": 0.00, "sell": 0.00}}
"question": "절반 매도", "answer": {{"buy": 0.00, "sell": 0.50}}
    
question: {prompt_text}, "answer": """
    )

    chain = template | llm | StrOutputParser()
    try:
        result = chain.invoke({"prompt_text": prompt_text})
        print("=================result=====================")
        print(result)
        print("============================================")

        result = json.loads(result)
        buy_ratio = result.get("buy")
        sell_ratio = result.get("sell")

        if buy_ratio is not None and sell_ratio is not None:
            return buy_ratio, sell_ratio
        else:
            logging.error(f"JSON 결과에 'buy' 또는 'sell' 키가 누락되었습니다: {result}")
            return 0.3, 0.3

    except Exception as e:
        logging.exception(f"정상적인 json 답변이 아니거나, 예상치 못한 오류 발생: {e}")
        return 0.3, 0.3

    # result = await chain.ainvoke({"prompt_text": prompt_text})
    # print("result" , result)
    # try:
    #     result = json.loads(result)
    # except JSONDecodeError:
    #     match = re.search(r"\{.*\}", result, re.DOTALL)
    #     json_str = match.group(0)
    #     result = json_str.replace("'", '"')
    #     try:
    #         result = json.loads(result)
    #         print(f"추출된 데이터: {result}")
    #     except json.JSONDecodeError as e:
    #         print(f"JSON 디코딩 오류 발생: {e}")
    #         logging.exception(JSONDecodeError(f"정상적인 json 답변이 아닙니다: {result}"))
    #         return 0.3, 0.3
    # return result.get("buy"), result.get("sell")


