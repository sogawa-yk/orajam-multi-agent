"""
============================================================
API ベースのツール (API-based Tools)
============================================================
エージェントのコード内に実装されているが、
処理の実体は外部の API サーバーにある。
ツール関数は HTTP リクエストを送り、レスポンスを返すだけ。

  エージェント  ──HTTP GET──→  API サーバー (別プロセス/別コンテナ)
                ←─JSON応答──

特徴:
  - ツールの実装はエージェント側にあるが、データ取得は外部に依存
  - ネットワーク通信が発生する
  - API サーバーが停止していると使えない
  - REST API, GraphQL など標準的なプロトコルを使用
============================================================
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests as http_client
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from shared import get_llm, print_header, print_step

# API サーバーの URL (docker-compose で定義)
API_BASE_URL = os.getenv("API_BASE_URL", "http://api-server:8000")

# ----------------------------------------------------------
# API ベースのツール: HTTP リクエストで外部 API を呼び出す
# ----------------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """都市名を指定して天気情報を取得する。外部の天気APIサーバーにHTTPリクエストを送る。"""
    # ★ ポイント: HTTP GET で外部 API を呼び出している
    try:
        response = http_client.get(f"{API_BASE_URL}/weather", params={"city": city})
        data = response.json()
        if "error" in data:
            return data["error"]
        return f"{data['city']}: 天気={data['weather']}, 気温={data['temperature']}℃, 湿度={data['humidity']}%"
    except http_client.ConnectionError:
        return "エラー: API サーバーに接続できません。"


@tool
def calculator(expression: str) -> str:
    """数式を計算する。例: '28 >= 25', '28 - 25'"""
    try:
        allowed = {"__builtins__": {}, "math": math}
        result = eval(expression, allowed)
        return str(result)
    except Exception as e:
        return f"計算エラー: {e}"


tools = [get_weather, calculator]

# ----------------------------------------------------------
# エージェント実行
# ----------------------------------------------------------

def run(question: str) -> str:
    llm = get_llm()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="あなたは天気情報アシスタントです。ツールを使って質問に答えてください。日本語で回答してください。",
    )

    print_header("API ベースツール エージェント実行")
    print(f"  API サーバー: {API_BASE_URL}")

    step_count = 0
    final_answer = ""

    for event in agent.stream({"messages": [("user", question)]}):
        for node_name, node_output in event.items():
            for msg in node_output.get("messages", []):
                if node_name == "agent":
                    step_count += 1
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print_step(
                                f"ステップ {step_count}: API ツール呼び出し",
                                f"ツール: {tc['name']}({tc['args']})\n"
                                f"※ HTTP GET {API_BASE_URL}/weather で外部 API を呼び出し",
                            )
                    else:
                        final_answer = msg.content
                        print_step(f"ステップ {step_count}: 最終回答", msg.content)
                elif node_name == "tools":
                    print_step("結果 (API レスポンス)", msg.content)

    return final_answer


if __name__ == "__main__":
    question = "東京の天気と気温を教えてください。気温が25度以上なら「暑い」、未満なら「過ごしやすい」と判断してください。"

    print_header("API ベースツール デモ")
    print(f"\n質問: {question}")
    print("\n[ツールの実装場所: エージェントのコード内 (HTTP で外部 API を呼び出す)]")

    answer = run(question)

    print_header("最終回答")
    print(answer)
