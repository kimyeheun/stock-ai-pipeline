from dsl import *
import re

INDICATOR_FUNC_MAP = {
    "RSI": {
        "func": "talib.RSI",
        "params": {"timeperiod": 14},
        "input": ["close"]
    },
    "BBANDS": {
        "func": "talib.BBANDS",
        "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
        "input": ["close"]
    },
    "MACD": {
        "func": "talib.MACD",
        "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        "input": ["close"]
    },
    "MOM": {
        "func": "talib.MOM",
        "params": {"timeperiod": 10},
        "input": ["close"]
    },
    "CCI": {
        "func": "talib.CCI",
        "params": {"timeperiod": 14},
        "input": ["high", "low", "close"]
    },
    "SMA": {
        "func": "talib.SMA",
        "params": {"timeperiod": 20},  # 기본값 (override 가능)
        "input": ["close"]
    },
    "EMA": {
        "func": "talib.EMA",
        "params": {"timeperiod": 20},
        "input": ["close"]
    },
    # 캔들패턴
    "CDLDOJI": {"func": "talib.CDLDOJI", "input": ["open", "high", "low", "close"]},
    "CDLHAMMER": {"func": "talib.CDLHAMMER", "input": ["open", "high", "low", "close"]},
    "CDLENGULFING": {"func": "talib.CDLENGULFING", "input": ["open", "high", "low", "close"]},
    "CDLSHOOTINGSTAR": {"func": "talib.CDLSHOOTINGSTAR", "input": ["open", "high", "low", "close"]},
    "CDLSPINNINGTOP": {"func": "talib.CDLSPINNINGTOP", "input": ["open", "high", "low", "close"]},
    "CDLMORNINGSTAR": {"func": "talib.CDLMORNINGSTAR", "input": ["open", "high", "low", "close"]},
    "CDLEVENINGSTAR": {"func": "talib.CDLEVENINGSTAR", "input": ["open", "high", "low", "close"]},
}

def indicator_to_code(indicator, params=None, var_prefix=""):
    # SMA50 등으로 들어오면 파싱
    base_indicator = indicator
    local_params = params or {}

    # SMA50, EMA200 등 파라미터 포함 이름 자동 해석
    match = re.match(r'^(SMA|EMA|WMA|DEMA|TEMA|TRIMA|KAMA|MAMA|T3)(\d+)$', indicator, re.IGNORECASE)
    if match:
        base_indicator = match.group(1).upper()
        timeperiod = int(match.group(2))
        local_params = {**local_params, "timeperiod": timeperiod}

    if base_indicator not in INDICATOR_FUNC_MAP:
        raise ValueError(f"Unknown indicator: {indicator}")

    m = INDICATOR_FUNC_MAP[base_indicator]
    func = m["func"]
    inputs = m["input"]
    param_str = ""
    all_params = m.get("params", {}).copy()
    all_params.update(local_params)
    if all_params:
        param_str = ", ".join([f"{k}={repr(v)}" for k, v in all_params.items()])
    input_args = ", ".join([f"df['{col}']" for col in inputs])
    if param_str:
        return f"{var_prefix}{indicator.lower()} = {func}({input_args}, {param_str})"
    else:
        return f"{var_prefix}{indicator.lower()} = {func}({input_args})"

def dsl_to_code(dsl: StrategyDSL, df_var='df') -> str:
    code_lines = []
    computed_indicators = set()

    def traverse_condition(cond, prefix=""):
        # 복합 논리 (AND/OR)
        if cond.logic and cond.conditions:
            inner = [traverse_condition(c) for c in cond.conditions]
            op = "&" if cond.logic == "AND" else "|"
            return f"({op.join(inner)})"
        # 단일조건
        ind = cond.indicator
        op = cond.operator
        val = cond.value
        compare = cond.compare_to
        params = cond.params or {}
        trend = cond.trend

        # 지표 선언 (중복 생성 방지)
        var_name = ind.lower()
        if ind not in computed_indicators:
            code_lines.append(indicator_to_code(ind, params))
            computed_indicators.add(ind)

        # 캔들패턴 처리
        if ind.startswith("CDL"):
            if op: # 예: == 100, == -100
                signal = f"({var_name} {op} {val})" if val is not None else f"({var_name} == 100)"
            else: # operator 없으면 100 신호 기준
                signal = f"({var_name} == 100)"
        else:
            signal = None
            if op and not trend:
                # 골든 크로스
                if op == "crosses_above" and compare:
                    if compare.lower() not in computed_indicators:
                        code_lines.append(indicator_to_code(compare))
                        computed_indicators.add(compare.lower())
                    signal = f"(({var_name}.shift(1) <= {compare.lower()}.shift(1)) & ({var_name} > {compare.lower()}))"
                # 데드 크로스
                elif op == "crosses_below" and compare:
                    if compare.lower() not in computed_indicators:
                        code_lines.append(indicator_to_code(compare))
                        computed_indicators.add(compare.lower())
                    signal = f"(({var_name}.shift(1) >= {compare.lower()}.shift(1)) & ({var_name} < {compare.lower()}))"
                elif op in ["<", ">", "<=", ">=", "==", "!="]:
                    rhs = compare.lower() if compare else val
                    signal = f"({var_name} {op} {rhs})"
                elif op == "is_trending_up":
                    signal = f"({var_name} > {var_name}.shift(1))"
                elif op == "is_trending_down":
                    signal = f"({var_name} < {var_name}.shift(1))"
                else:
                    signal = "1"
            elif trend and not op:
                if trend == "up":
                    trend_expr = f"({var_name} > {var_name}.shift(1))"
                elif trend == "down":
                    trend_expr = f"({var_name} < {var_name}.shift(1))"
                else:
                    trend_expr = "1"
                # 기존 조건과 trend 조건을 AND(&)로 결합
                signal = f"({signal} & {trend_expr})"
                # 3. operator + trend 모두 있는 경우
            elif trend and op:
                if op == "crosses_above" and compare:
                    if compare.lower() not in computed_indicators:
                        code_lines.append(indicator_to_code(compare))
                        computed_indicators.add(compare.lower())
                    op_signal = f"(({var_name}.shift(1) <= {compare.lower()}.shift(1)) & ({var_name} > {compare.lower()}))"
                elif op == "crosses_below" and compare:
                    if compare.lower() not in computed_indicators:
                        code_lines.append(indicator_to_code(compare))
                        computed_indicators.add(compare.lower())
                    op_signal = f"(({var_name}.shift(1) >= {compare.lower()}.shift(1)) & ({var_name} < {compare.lower()}))"
                elif op in ["<", ">", "<=", ">=", "==", "!="]:
                    rhs = compare.lower() if compare else val
                    op_signal = f"({var_name} {op} {rhs})"
                elif op == "is_trending_up":
                    op_signal = f"({var_name} > {var_name}.shift(1))"
                elif op == "is_trending_down":
                    op_signal = f"({var_name} < {var_name}.shift(1))"
                else:
                    op_signal = "1"
                # trend 코드
                if trend == "up":
                    trend_expr = f"({var_name} > {var_name}.shift(1))"
                elif trend == "down":
                    trend_expr = f"({var_name} < {var_name}.shift(1))"
                else:
                    trend_expr = "1"
                signal = f"({op_signal} & {trend_expr})"
            # 4. operator, trend 둘 다 없는 경우 (DSL 불완전)
            else:
                signal = "1"

        return signal

    entry_signals = []
    for i, entry in enumerate(dsl.entries):
        cond_expr = traverse_condition(entry.root_condition)
        signal_var = f"entry_signal_{i}"
        code_lines.append(f"{signal_var} = {cond_expr}")
        entry_signals.append(signal_var)

    exit_signals = []
    for i, exit in enumerate(dsl.exits):
        cond_expr = traverse_condition(exit.root_condition)
        signal_var = f"exit_signal_{i}"
        code_lines.append(f"{signal_var} = {cond_expr}")
        exit_signals.append(signal_var)

    # 최종 시그널 변수 생성
    if entry_signals:
        code_lines.append(
            f"final_buy_signal = {' | '.join(entry_signals)}"
            if len(entry_signals) > 1 else
            f"final_buy_signal = {entry_signals[0]}"
        )
    else:
        code_lines.append(f"final_buy_signal = pd.Series([False]*len({df_var}), index={df_var}.index)")

    if exit_signals:
        code_lines.append(
            f"final_sell_signal = {' | '.join(exit_signals)}"
            if len(exit_signals) > 1 else
            f"final_sell_signal = {exit_signals[0]}"
        )
    else:
        code_lines.append(f"final_sell_signal = pd.Series([False]*len({df_var}), index={df_var}.index)")

    return "\n".join(code_lines)

