from dsl import *

SYSTEM_PROMPT = """
당신은 금융 및 알고리즘 트레이딩 전략 전문가입니다.
사용자가 입력하는 주식 매매 전략을 아래 JSON 구조의 DSL로 변환하세요.
반드시 아래의 JSON 규칙을 따르세요.
- 조건은 논리적으로 중첩(AND/OR)할 수 있습니다.
- 각 신호(매수, 매도)마다 position_size(운용비중)를 필수로 입력하세요.

[예시 입력]
RSI가 30 밑에서 반등하거나, SMA50이 SMA200을 상향 돌파하면 50% 매수. RSI가 70 이상이면 전량 매도.

[예시 출력]
{
  "strategy_title": "RSI+SMA 복합 매매",
  "entries": [
    {
      "logic": "OR",
      "conditions": [
        {
          "logic": "AND",
          "conditions": [
            {
              "indicator": "RSI",
              "operator": "<",
              "value": 30,
              "trend": "up"
            }
          ]
        },
        {
          "logic": "AND",
          "conditions": [
            {
              "indicator": "SMA",
              "params": {"timeperiod": 50},
              "operator": "crosses_above",
              "compare_to": "SMA200"
            }
          ]
        }
      ],
      "action": "buy",
      "position_size": 0.5
    }
  ],
  "exits": [
    {
      "logic": "AND",
      "conditions": [
        {
          "indicator": "RSI",
          "operator": ">=",
          "value": 70
        }
      ],
      "action": "sell",
      "position_size": 1.0
    }
  ]
}
"""

function_schema = {
    "name": "parse_trading_strategy",
    "description": "자연어 매매 전략을 복합 논리 및 운용 비중을 포함한 DSL로 변환",
    "parameters": {
        "type": "object",
        "properties": {
            "strategy_title": {"type": "string"},
            "entries": {
                "type": "array",
                "items": {"$ref": "#/definitions/EntryOrExit"}
            },
            "exits": {
                "type": "array",
                "items": {"$ref": "#/definitions/EntryOrExit"}
            }
        },
        "required": ["strategy_title", "entries", "exits"],
        "definitions": {
            "EntryOrExit": {
                "type": "object",
                "properties": {
                    "logic": {
                        "type": "string",
                        "enum": ["AND", "OR"]
                    },
                    "conditions": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"$ref": "#/definitions/Condition"},
                                {"$ref": "#/definitions/EntryOrExit"}  # 복합 논리 재귀
                            ]
                        }
                    },
                    "action": {"type": "string", "enum": ["buy", "sell", "buy_partial", "sell_partial", "hold"]},
                    "position_size": {"type": "number"},
                },
                "required": ["logic", "conditions", "action", "position_size"]
            },
            "Condition": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string"},
                    "operator": {"type": "string"},
                    "value": {"type": ["number", "string", "null"]},
                    "compare_to": {"type": ["string", "null"]},
                    "params": {"type": "object"},
                    "trend": {"type": ["string", "null"]},
                    "lag": {"type": "integer"}
                },
                "required": ["indicator", "operator"]
            }
        }
    }
}

import json
from typing import Any, Dict, List
# 위에서 정의한 DSL 클래스(StrategyDSL, EntryOrExit, Condition)는 이미 import했다고 가정

def parse_condition(cond_json: Dict[str, Any]) -> Condition:
    """
    단일 Condition 또는 AND/OR 복합 조건(재귀) JSON을 파싱하여 Condition 객체로 변환.
    """
    # 복합 논리 조건인지 확인 (logic 필드 유무)
    if 'logic' in cond_json and 'conditions' in cond_json:
        # 하위 conditions도 재귀적으로 변환
        sub_conditions = [parse_condition(sub) for sub in cond_json['conditions']]
        return Condition(
            logic=cond_json['logic'],
            conditions=sub_conditions
        )
    else:
        # 단일 leaf 조건
        return Condition(
            indicator=cond_json.get('indicator'),
            operator=cond_json.get('operator'),
            value=cond_json.get('value'),
            compare_to=cond_json.get('compare_to'),
            params=cond_json.get('params'),
            trend=cond_json.get('trend'),
            lag=cond_json.get('lag', 0)
        )

def parse_entry_or_exit(entry_json: Dict[str, Any]) -> EntryOrExit:
    """
    Entry/Exit 조건(복합 논리 + 액션 + position_size) JSON을 파싱하여 EntryOrExit 객체로 변환.
    """
    # 조건부 복합 논리 root_condition 파싱
    root_condition = parse_condition({
        "logic": entry_json.get("logic"),
        "conditions": entry_json.get("conditions", [])
    })
    return EntryOrExit(
        root_condition=root_condition,
        action=entry_json["action"],
        position_size=entry_json.get("position_size", 1.0),
        comment=entry_json.get("comment")
    )

def parse_strategy_dsl(dsl_json: Dict[str, Any]) -> StrategyDSL:
    """
    LLM이 반환한 최상위 DSL JSON을 전체 StrategyDSL 객체로 변환.
    """
    entries = [parse_entry_or_exit(entry) for entry in dsl_json.get("entries", [])]
    exits = [parse_entry_or_exit(exit_) for exit_ in dsl_json.get("exits", [])]
    return StrategyDSL(
        strategy_title=dsl_json.get("strategy_title", ""),
        entries=entries,
        exits=exits,
        custom_logic_required=dsl_json.get("custom_logic_required", False),
        custom_logic_description=dsl_json.get("custom_logic_description")
    )


async def generate_dsl(client, natural_text: str) -> StrategyDSL:
    response = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": f"{SYSTEM_PROMPT}"},
            {"role": "user", "content": f"{natural_text}"}
        ],
        functions=[function_schema],
        function_call={"name": "parse_trading_strategy"}
    )

    function_args_str = response.choices[0].message.function_call.arguments

    dsl_json = json.loads(function_args_str)

    # 최종 DSL 객체로 변환
    dsl_obj = parse_strategy_dsl(dsl_json)

    return dsl_obj
