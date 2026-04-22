"""
============================================================
MCP ツール (Model Context Protocol)
============================================================
ツールがエージェントの「外」(別プロセス) に実装されている。
エージェントは MCP プロトコルでサーバーに接続し、
ツールを動的に発見して利用する。

  エージェント  ──MCP (stdio)──→  MCP サーバー (別プロセス)
                ←──ツール一覧──
                ──ツール呼出──→
                ←──実行結果────

特徴:
  - ツールの実装がエージェントの外にある
  - エージェントは MCP プロトコルでツールを「発見」する
  - サーバーを差し替えるだけでツールを追加・変更できる
  - 言語に依存しない (Python以外でMCPサーバーを書ける)
============================================================
"""

import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from shared import get_llm, print_header, print_step


async def run(question: str) -> str:
    """MCP サーバーに接続し、ツールを取得してエージェントを実行する"""
    llm = get_llm()

    # =========================================================
    # Step 1: MCP サーバーをサブプロセスとして起動し、接続する
    # =========================================================
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
    server_params = StdioServerParameters(
        command="python",
        args=[server_script],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # =========================================================
            # Step 2: MCP ツールを LangChain ツールとして読み込む
            #         load_mcp_tools がツール発見 + 変換を一括で行う
            # =========================================================
            tools = await load_mcp_tools(session)

            print_step(
                "MCP ツール発見",
                "MCP サーバーに接続し、利用可能なツールを取得しました:\n"
                + "\n".join(f"  - {t.name}: {t.description}" for t in tools),
            )

            # =========================================================
            # Step 3: エージェント実行
            # =========================================================
            agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt="あなたは天気情報アシスタントです。ツールを使って質問に答えてください。日本語で回答してください。",
            )

            print_header("MCP ツール エージェント実行")

            step_count = 0
            final_answer = ""

            async for event in agent.astream({"messages": [("user", question)]}):
                for node_name, node_output in event.items():
                    for msg in node_output.get("messages", []):
                        if node_name == "agent":
                            step_count += 1
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print_step(
                                        f"ステップ {step_count}: MCP ツール呼び出し",
                                        f"ツール: {tc['name']}({tc['args']})\n"
                                        f"※ MCP プロトコルで外部サーバーのツールを呼び出し",
                                    )
                            else:
                                final_answer = msg.content
                                print_step(f"ステップ {step_count}: 最終回答", msg.content)
                        elif node_name == "tools":
                            print_step("結果 (MCP サーバーからの応答)", msg.content)

            return final_answer


if __name__ == "__main__":
    question = "東京の天気と気温を教えてください。気温が25度以上なら「暑い」、未満なら「過ごしやすい」と判断してください。"

    print_header("MCP ツール デモ")
    print(f"\n質問: {question}")
    print("\n[ツールの実装場所: MCP サーバー (別プロセス)]")

    answer = asyncio.run(run(question))

    print_header("最終回答")
    print(answer)
