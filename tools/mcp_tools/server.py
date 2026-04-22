"""
============================================================
MCP サーバー (Model Context Protocol)
============================================================
エージェントの「外」に実装されたツール。
MCP プロトコル (stdio) でエージェントと通信する。

エージェントは:
  1. MCP サーバーに接続する
  2. 利用可能なツール一覧を取得する (ツール発見)
  3. ツールを呼び出し、結果を受け取る

ローカルツールや API ツールとの違い:
  - ツールの定義・実装がエージェントの外にある
  - エージェントは事前にツールの実装を知らなくてよい
  - MCP プロトコルでツールを動的に発見・利用できる
============================================================
"""

from mcp.server.fastmcp import FastMCP

# MCP サーバーを作成 (stdio トランスポート)
mcp = FastMCP("天気情報サーバー")

# 天気データ (ローカルツール・API と同じデータ)
WEATHER_DATA = {
    "東京": {"city": "東京", "weather": "晴れ", "temperature": 28, "humidity": 55},
    "大阪": {"city": "大阪", "weather": "曇り", "temperature": 24, "humidity": 65},
    "札幌": {"city": "札幌", "weather": "雨",   "temperature": 18, "humidity": 80},
}


@mcp.tool()
def get_weather(city: str) -> str:
    """都市名を指定して天気情報を取得する。"""
    data = WEATHER_DATA.get(city)
    if not data:
        return f"'{city}' の天気データは見つかりませんでした。"
    return f"{data['city']}: 天気={data['weather']}, 気温={data['temperature']}℃, 湿度={data['humidity']}%"


@mcp.tool()
def calculator(expression: str) -> str:
    """数式を計算する。例: '28 >= 25', '28 - 25'"""
    import math
    try:
        allowed = {"__builtins__": {}, "math": math}
        result = eval(expression, allowed)
        return str(result)
    except Exception as e:
        return f"計算エラー: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
