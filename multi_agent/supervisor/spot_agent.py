"""
観光スポットエージェント (A2A サーバー) — スーパーバイザー型ワーカー
内部は ReAct エージェントで、観光スポットツールを持つ。
"""

import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from python_a2a import A2AServer, AgentCard, AgentSkill, MessageRole, create_text_message, run_server
from shared import get_llm, SPOT_DATA

# ---- ツール ----

@tool
def get_spots(city: str) -> str:
    """都市の観光スポット一覧を取得する"""
    spots = SPOT_DATA.get(city)
    if not spots:
        return f"'{city}' の観光スポットデータはありません。"
    lines = [f"- {s['name']} ({s['category']}): {s['description']}" for s in spots]
    return f"{city}の観光スポット:\n" + "\n".join(lines)

# ---- A2A サーバー ----

class SpotAgentServer(A2AServer):
    def __init__(self):
        card = AgentCard(
            name="Spot Agent",
            description="観光スポット情報を提供するエージェント",
            url="http://localhost:5002",
            skills=[AgentSkill(name="spots", description="都市の観光スポットを検索・回答する")],
        )
        super().__init__(agent_card=card)
        self.react_agent = create_react_agent(
            model=get_llm(),
            tools=[get_spots],
            prompt="あなたは観光スポットの専門家です。ツールを使って正確に回答してください。日本語で回答してください。",
        )

    def handle_message(self, message):
        trace, answer, pending = [], "", {}

        for event in self.react_agent.stream({"messages": [("user", message.content.text)]}):
            for node_name, node_output in event.items():
                for msg in node_output.get("messages", []):
                    if node_name == "agent" and hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            pending[tc["id"]] = tc
                    elif node_name == "tools":
                        tc = pending.pop(getattr(msg, "tool_call_id", None), None)
                        if tc:
                            trace.append({
                                "agent": "Spot Agent", "type": "tool",
                                "name": tc["name"], "args": tc["args"],
                                "result": msg.content if isinstance(msg.content, str) else str(msg.content),
                            })
                    elif node_name == "agent" and msg.content:
                        answer = msg.content

        response = json.dumps({"trace": trace, "answer": answer}, ensure_ascii=False)
        return create_text_message(response, role=MessageRole.AGENT)


if __name__ == "__main__":
    print("Spot Agent starting on port 5002...")
    run_server(SpotAgentServer(), host="0.0.0.0", port=5002)
