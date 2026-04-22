"""
============================================================
ローカルツール (Local Tools)
============================================================
エージェントのコード内に直接実装されているツール。
外部との通信は一切なく、プロセス内で完結する。

例:
  - 計算処理 (calculator)
  - ローカルファイルの読み取り (read_file)
  - データの変換・加工

特徴:
  - 高速 (ネットワーク遅延なし)
  - 外部依存なし (API障害の影響を受けない)
  - エージェントと同じプロセスで動作する
============================================================
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from shared import get_llm, print_header, print_step, WEATHER_DATA

# ----------------------------------------------------------
# ローカルツール: すべてエージェントのコード内に実装
# ----------------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """都市名を指定して天気情報を取得する。データはプログラム内に定義されている。"""
    # ★ ポイント: データがコード内の辞書に直接定義されている
    data = WEATHER_DATA.get(city)
    if not data:
        return f"'{city}' の天気データは見つかりませんでした。"
    return f"{data['city']}: 天気={data['weather']}, 気温={data['temperature']}℃, 湿度={data['humidity']}%"


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

    print_header("ローカルツール エージェント実行")

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
                                f"ステップ {step_count}: ローカルツール呼び出し",
                                f"ツール: {tc['name']}({tc['args']})\n"
                                f"※ プロセス内の関数を直接呼び出し (ネットワーク通信なし)",
                            )
                    else:
                        final_answer = msg.content
                        print_step(f"ステップ {step_count}: 最終回答", msg.content)
                elif node_name == "tools":
                    print_step("結果", msg.content)

    return final_answer


if __name__ == "__main__":
    question = "東京の天気と気温を教えてください。気温が25度以上なら「暑い」、未満なら「過ごしやすい」と判断してください。"

    print_header("ローカルツール デモ")
    print(f"\n質問: {question}")
    print("\n[ツールの実装場所: エージェントと同じコード内]")

    answer = run(question)

    print_header("最終回答")
    print(answer)
