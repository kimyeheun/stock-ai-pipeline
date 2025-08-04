# NOTE: 프롬프트 분기점
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_huggingface import HuggingFacePipeline

from api.ai.prompt_interpreter import professional_level


# 상 중 하 분류 후 각 코드로 분기
async def classify_difficulty(prompt_text: str) -> str:
    # hf_model = pipeline("text-generation", model="deepseek-ai/DeepSeek-V3")
    # llm = HuggingFaceLLM(pipeline=hf_model)
    llm = HuggingFacePipeline.from_model_id(
        model_id="deepseek-ai/deepseek-llm-7b-base",
        task="text-generation",
        device_map="cuda"
    )
    # 프롬프트 설정 및 파이프라인 실행
    template = PromptTemplate(input_variables=["prompt_text"],
                              template="""아래 주식 전략을
1: 감각적, 지표 없음
2: 지표 언급
3: 복합 조건/트레이딩 로직
중 하나로 분류. 숫자만 답해
문장: {prompt_text}""")
    story_chain = LLMChain(llm=llm, prompt_template=template)

    # 파이프라인 실행
    result = story_chain.run(prompt_text=prompt_text)

    # 여러 체인을 연결한 확장 가능한 파이프라인
    # chain1 = LLMChain(llm=llm, prompt_template=PromptTemplate.from_template("Describe {topic}"))
    # chain2 = LLMChain(llm=llm, prompt_template=PromptTemplate.from_template("Summarize the description of {topic}"))
    # # 순차적으로 체인 연결
    # sequential_chain = SequentialChain(chains=[chain1, chain2])
    print(result)
    return result


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

