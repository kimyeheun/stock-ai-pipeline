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
        "params": {"timeperiod": 20},
        "input": ["close"]
    },
    "EMA": {
        "func": "talib.EMA",
        "params": {"timeperiod": 20},
        "input": ["close"]
    },
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

    if base_indicator == "MACD":
        return f"macd_line, macd_signal, macd_hist = {func}({input_args}, {param_str})"
    if param_str:
        return f"{var_prefix}{indicator.lower()} = {func}({input_args}, {param_str})"
    else:
        return f"{var_prefix}{indicator.lower()} = {func}({input_args})"

def dsl_to_code(dsl: StrategyDSL, df_var='df') -> str:
    code_lines = []
    computed_indicators = set()

    def traverse_condition(cond, prefix=""):
        if cond.logic and cond.conditions:
            inner = [traverse_condition(c) for c in cond.conditions]
            op = "&" if cond.logic == "AND" else "|"
            return f"({op.join(inner)})"

        ind = cond.indicator
        op = cond.operator
        val = cond.value
        compare = cond.compare_to
        params = cond.params or {}
        trend = cond.trend

        var_name = ind.lower()
        if ind not in computed_indicators:
            code_lines.append(indicator_to_code(ind, params))
            computed_indicators.add(ind)

        if compare and compare.lower() not in computed_indicators:
            code_lines.append(indicator_to_code(compare))
            computed_indicators.add(compare.lower())

        if ind.startswith("CDL"):
            return f"({var_name} {op} {val})" if op else f"({var_name} == 100)"

        if ind == "MACD" and op == "rising":
            return f"(macd_line > macd_signal)"
        if ind == "MACD" and op == "trend_change" and trend == "down":
            if ind not in computed_indicators:
                code_lines.append(
                    "macd_line, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)")
                computed_indicators.add(ind)
            return "(macd_line< macd.shift(1)) & (macd_line < macd_signal)"

        base = None
        if op in ["<", ">", "<=", ">=", "==", "!="]:
            base = f"({var_name} {op} {val})"
        elif op == "crosses_above" and compare:
            base = f"(({var_name}.shift(1) <= {compare.lower()}.shift(1)) & ({var_name} > {compare.lower()}))"
        elif op == "crosses_below" and compare:
            base = f"(({var_name}.shift(1) >= {compare.lower()}.shift(1)) & ({var_name} < {compare.lower()}))"
        elif op == "is_trending_up":
            base = f"({var_name} > {var_name}.shift(1))"
        elif op == "is_trending_down":
            base = f"({var_name} < {var_name}.shift(1))"
        else:
            base = "1"

        if trend == "up":
            trend_expr = f"({var_name} > {var_name}.shift(1))"
            return f"({base} & {trend_expr})"
        elif trend == "down":
            trend_expr = f"({var_name} < {var_name}.shift(1))"
            return f"({base} & {trend_expr})"

        return base

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

    code_lines.append(
        f"final_buy_signal = {' | '.join(entry_signals)}" if entry_signals else f"final_buy_signal = pd.Series([False]*len({df_var}), index={df_var}.index)"
    )
    code_lines.append(
        f"final_sell_signal = {' | '.join(exit_signals)}" if exit_signals else f"final_sell_signal = pd.Series([False]*len({df_var}), index={df_var}.index)"
    )

    return "\n".join(code_lines)
