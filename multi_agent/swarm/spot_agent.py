"""
観光スポットエージェント (A2A サーバー) — スウォーム型
自分で処理できない部分は、自律的に他のエージェントにハンドオフする。
"""

import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from python_a2a import A2AServer, A2AClient, AgentCard, AgentSkill, MessageRole, create_text_message, run_server
from shared import get_llm, SPOT_DATA

# collect_trace を共有 (weather_agent と同じロジック)
from weather_agent import collect_trace

WEATHER_AGENT_URL = os.getenv("WEATHER_AGENT_URL", "http://sw-weather-agent:5001")

# ---- ツール (観光 + ハンドオフ) ----

@tool
def get_spots(city: str) -> str:
    """都市の観光スポット一覧を取得する"""
    spots = SPOT_DATA.get(city)
    if not spots:
        return f"'{city}' の観光スポットデータはありません。"
    lines = [f"- {s['name']} ({s['category']}): {s['description']}" for s in spots]
    return f"{city}の観光スポット:\n" + "\n".join(lines)

@tool
def handoff_to_weather_agent(query: str) -> str:
    """天気情報が必要なとき、天気エージェントに委譲する (A2A ハンドオフ)"""
    client = A2AClient(WEATHER_AGENT_URL)
    return client.ask(query)

# ---- A2A サーバー ----

class SpotAgentServer(A2AServer):
    def __init__(self):
        card = AgentCard(
            name="Spot Agent (Swarm)",
            description="観光スポット情報を提供し、必要に応じて他のエージェントにハンドオフするエージェント",
            url="http://localhost:5002",
            skills=[
                AgentSkill(name="spots", description="観光スポットの検索"),
                AgentSkill(name="handoff", description="天気エージェントへのハンドオフ"),
            ],
        )
        super().__init__(agent_card=card)
        self.react_agent = create_react_agent(
            model=get_llm(),
            tools=[get_spots, handoff_to_weather_agent],
            prompt=(
                "あなたは観光スポットの専門家です。観光スポットに関する質問にはツールを使って回答してください。"
                "天気に関する質問が含まれている場合は、handoff_to_weather_agent を使って天気エージェントに委譲してください。"
                "すべての情報が揃ったら、最終的な回答をまとめてください。"
                "日本語で回答してください。"
            ),
        )

    def handle_message(self, message):
        trace, answer = collect_trace(
            self.react_agent, message.content.text,
            agent_name="Spot Agent",
            handoff_tool_name="handoff_to_weather_agent",
            handoff_target="Weather Agent",
            handoff_url=WEATHER_AGENT_URL,
        )
        response = json.dumps({"trace": trace, "answer": answer}, ensure_ascii=False)
        return create_text_message(response, role=MessageRole.AGENT)


if __name__ == "__main__":
    print("Spot Agent (Swarm) starting on port 5002...")
    run_server(SpotAgentServer(), host="0.0.0.0", port=5002)
