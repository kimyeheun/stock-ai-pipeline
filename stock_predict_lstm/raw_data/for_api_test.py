import pandas as pd
import json

# 1. CSV 파일 로드
df = pd.read_csv('stock_data_from_api_kakao.csv')

# 혹시 컬럼명이 다를 경우 직접 매핑
col_map = {
    'mkp': 'open',
    'hipr': 'high',
    'lopr': 'low',
    'clpr': 'close',
    'trqu': 'volume'
}

df = df[list(col_map.keys())].rename(columns=col_map)

# 3. JSON 구조로 만들기
room_id = 0  # 원하는 roomId로 설정
json_data = {
    "roomId": room_id,
    "ohlcv": {
        "open": df['open'].tolist(),
        "high": df['high'].tolist(),
        "low": df['low'].tolist(),
        "close": df['close'].tolist(),
        "volume": df['volume'].tolist(),
    }
}

# 4. JSON 파일로 저장 (테스트용)
with open('init_request.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

# 5. print로도 확인 가능
print(json.dumps(json_data, ensure_ascii=False, indent=2))
