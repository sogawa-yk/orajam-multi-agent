"""
天気エージェント (A2A サーバー) — スーパーバイザー型ワーカー
内部は ReAct エージェントで、天気ツールを持つ。
"""

import sys, os, math, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from python_a2a import A2AServer, AgentCard, AgentSkill, MessageRole, create_text_message, run_server
from shared import get_llm, WEATHER_DATA

# ---- ツール ----

@tool
def get_weather(city: str) -> str:
    """都市の天気・気温・湿度を取得する"""
    data = WEATHER_DATA.get(city)
    if not data:
        return f"'{city}' の天気データはありません。"
    return f"{data['city']}: 天気={data['weather']}, 気温={data['temperature']}℃, 湿度={data['humidity']}%"

@tool
def calculator(expression: str) -> str:
    """数式を計算する"""
    try:
        return str(eval(expression, {"__builtins__": {}, "math": math}))
    except Exception as e:
        return f"計算エラー: {e}"

# ---- A2A サーバー ----

class WeatherAgentServer(A2AServer):
    def __init__(self):
        card = AgentCard(
            name="Weather Agent",
            description="天気情報を提供するエージェント",
            url="http://localhost:5001",
            skills=[AgentSkill(name="weather", description="都市の天気情報を検索・回答する")],
        )
        super().__init__(agent_card=card)
        self.react_agent = create_react_agent(
            model=get_llm(),
            tools=[get_weather, calculator],
            prompt="あなたは天気情報の専門家です。ツールを使って正確に回答してください。日本語で回答してください。",
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
                                "agent": "Weather Agent", "type": "tool",
                                "name": tc["name"], "args": tc["args"],
                                "result": msg.content if isinstance(msg.content, str) else str(msg.content),
                            })
                    elif node_name == "agent" and msg.content:
                        answer = msg.content

        response = json.dumps({"trace": trace, "answer": answer}, ensure_ascii=False)
        return create_text_message(response, role=MessageRole.AGENT)


if __name__ == "__main__":
    print("Weather Agent starting on port 5001...")
    run_server(WeatherAgentServer(), host="0.0.0.0", port=5001)
