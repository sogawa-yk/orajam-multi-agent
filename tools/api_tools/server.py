"""
============================================================
天気 API サーバー (FastAPI)
============================================================
エージェントから HTTP リクエストで呼び出される外部 API。
エージェントとは別プロセス (別コンテナ) で動作する。

エンドポイント:
  GET /weather?city=東京  → 天気情報をJSONで返す
  GET /health             → ヘルスチェック
============================================================
"""

from fastapi import FastAPI, Query

app = FastAPI(title="天気 API サーバー")

# 天気データ (ローカルツールと同じデータ)
WEATHER_DATA = {
    "東京": {"city": "東京", "weather": "晴れ", "temperature": 28, "humidity": 55},
    "大阪": {"city": "大阪", "weather": "曇り", "temperature": 24, "humidity": 65},
    "札幌": {"city": "札幌", "weather": "雨",   "temperature": 18, "humidity": 80},
}


@app.get("/weather")
def get_weather(city: str = Query(..., description="都市名")):
    """都市の天気情報を返す"""
    data = WEATHER_DATA.get(city)
    if not data:
        return {"error": f"'{city}' の天気データは見つかりませんでした。"}
    return data


@app.get("/health")
def health():
    return {"status": "ok"}
