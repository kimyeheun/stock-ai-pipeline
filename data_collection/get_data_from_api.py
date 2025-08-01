import csv
import os
from dotenv import load_dotenv
import pandas as pd
import requests
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("API_KEY")
PAGE_NO = 12
RESULTS_DIR = "./results"

# 1. 종목 리스트 조회 및 (isinCd, 종목명) 매핑 저장
def fetch_stock_list(num_of_rows=100000):
    url = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"
    params = {
        "serviceKey": API_KEY,
        "numOfRows": num_of_rows,
        "pageNo": PAGE_NO,
        "resultType": "json",
    }
    response = requests.get(url, params=params)
    data = response.json()
    items = data['response']['body']['items']['item']

    stocks = [(item['isinCd'], item['itmsNm']) for item in items]
    return stocks

def save_stock_mapping(stocks, file_path):
    # 1. 기존 파일 읽기 (없으면 빈 리스트)
    existing_stocks = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_stocks.append((row["isinCd"], row["itmsNm"]))

    # 2. 기존 + 신규 합치기
    all_stocks = existing_stocks + stocks

    # 3. isinCd 기준 중복 제거
    unique_stocks = {}
    for isinCd, itmsNm in all_stocks:
        unique_stocks[isinCd] = itmsNm

    new_stocks = sorted(unique_stocks.items())

    # 4. 저장 (전체를 다시 저장, append가 아니라 전체 유니크로 overwrite)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["isinCd", "itmsNm"])
        for row in new_stocks:
            writer.writerow(row)

    print(f"총 저장 종목 수: {len(new_stocks)}")


def load_stock_mapping(file_path):
    stocks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stocks.append((row["isinCd"], row["itmsNm"]))
    return stocks

# 2. 개별 종목에 대해 전체 일자 차트 데이터 저장
def fetch_daily_chart_data(isinCd):
    url = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"
    all_data = []
    page_no = 1
    while True:
        params = {
            "serviceKey": API_KEY,
            "numOfRows": 100000,
            "pageNo": page_no,
            "isinCd": isinCd,
            "resultType": "json",
        }
        response = requests.get(url, params=params)
        data = response.json()
        try:
            items = data['response']['body']['items']['item']
        except Exception:
            break  # 더 이상 데이터 없음

        all_data.extend(items)
        # 페이지가 더 있는지 체크
        if len(items) < 100:
            break
        page_no += 1
    return all_data

def save_chart_data(isinCd, itmsNm, data):
    df = pd.DataFrame(data)
    filename = f"{isinCd}-{itmsNm}.csv"
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False, encoding='utf-8-sig')

# 3. 메인 실행 함수
def main():
    mapping_csv = os.path.join(RESULTS_DIR, "stock_mapping.csv")

    # 1. 종목리스트 조회 & 저장
    print("종목리스트 조회중...")
    stocks = fetch_stock_list()
    save_stock_mapping(stocks, mapping_csv)
    print(f"종목 매핑 CSV 저장 완료: {mapping_csv}")

    # 2. 저장된 종목명 CSV를 읽어 각 종목별 차트 데이터 저장
    stocks = load_stock_mapping(mapping_csv)

    for isinCd, itmsNm in tqdm(stocks, desc="전체 종목 진행률"):
        filename = f"{isinCd}-{itmsNm}.csv"
        path = os.path.join(RESULTS_DIR, filename)

        if os.path.exists(path):
            continue

        data = fetch_daily_chart_data(isinCd)
        if data:
            save_chart_data(isinCd, itmsNm, data)
        else:
            print(f"{itmsNm}→ 데이터 없음.")

if __name__ == "__main__":
    main()
