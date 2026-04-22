"""
============================================================
ReAct パターン (Reasoning + Acting) — LangGraph 実装
============================================================
LLMが「思考 → 行動 → 観察」のループを繰り返して問題を解決する。

グラフ構造:
  [START] → agent → should_continue? → tools → agent → ... → [END]

  - agent ノード : LLMが思考し、ツールを呼ぶか最終回答を出すか決める
  - tools ノード : LLMが選んだツールを実行し、結果を返す
  - 条件分岐     : ツール呼び出しがあれば tools へ、なければ END へ

LangGraph の create_react_agent で上記グラフを自動構築する。

参考: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
============================================================
"""

import sys, os, math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from shared import get_llm, print_header, print_step

# ----------------------------------------------------------
# ツール定義 (@tool デコレータで LangChain Tool 化)
# ----------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """数式を計算する。例: '2 + 3 * 4', '333 / 2'"""
    try:
        allowed = {"__builtins__": {}, "math": math}
        result = eval(expression, allowed)
        return str(result)
    except Exception as e:
        return f"計算エラー: {e}"


@tool
def search(query: str) -> str:
    """キーワードで情報を検索する。例: '東京タワー', '富士山'"""
    knowledge = {
        "東京タワー": "東京タワーは1958年に完成した高さ333mの電波塔です。",
        "富士山": "富士山は標高3,776mで日本最高峰の山です。",
        "python": "Pythonは1991年にGuido van Rossumが開発したプログラミング言語です。",
    }
    for key, value in knowledge.items():
        if key in query.lower() or key in query:
            return value
    return f"'{query}' に関する情報は見つかりませんでした。"


tools = [calculator, search]

# ----------------------------------------------------------
# ReAct エージェント (create_react_agent で構築)
# ----------------------------------------------------------

def run_react(question: str) -> str:
    """ReActエージェントを実行し、各ステップの思考をコンソール出力する"""
    llm = get_llm(temperature=0.2)

    # create_react_agent: agent ノードと tools ノードを持つグラフを自動生成
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="あなたは質問に答えるアシスタントです。必要に応じてツールを使ってください。日本語で回答してください。",
    )

    print_header("ReAct エージェント実行開始")

    # stream でステップごとのイベントを取得して表示
    step_count = 0
    final_answer = ""

    for event in agent.stream({"messages": [("user", question)]}):
        # event は {"agent": {...}} または {"tools": {...}} の形式
        for node_name, node_output in event.items():
            messages = node_output.get("messages", [])

            for msg in messages:
                if node_name == "agent":
                    step_count += 1

                    # ツール呼び出しがある場合 (思考 → 行動)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print_step(
                                f"ステップ {step_count}: 思考 → 行動",
                                f"思考: ツール '{tc['name']}' を使って調べる\n"
                                f"行動: {tc['name']}({tc['args']})",
                            )
                    else:
                        # 最終回答
                        final_answer = msg.content
                        print_step(
                            f"ステップ {step_count}: 最終回答",
                            msg.content,
                        )

                elif node_name == "tools":
                    # ツール実行結果 (観察)
                    print_step("観察 (ツール結果)", msg.content)

    return final_answer


# ----------------------------------------------------------
# 実行
# ----------------------------------------------------------

if __name__ == "__main__":
    question = "東京タワーと富士山の高さをそれぞれ調べて、2つの高さの差を計算し、わかりやすく説明してください。"

    print_header("ReAct パターン デモ (LangGraph)")
    print(f"\n質問: {question}")

    answer = run_react(question)

    print_header("最終回答")
    print(answer)
