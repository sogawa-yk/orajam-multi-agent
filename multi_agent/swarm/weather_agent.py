"""
天気エージェント (A2A サーバー) — スウォーム型
自分で処理できない部分は、自律的に他のエージェントにハンドオフする。
"""

import sys, os, math, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from python_a2a import A2AServer, A2AClient, AgentCard, AgentSkill, MessageRole, create_text_message, run_server
from shared import get_llm, WEATHER_DATA

SPOT_AGENT_URL = os.getenv("SPOT_AGENT_URL", "http://sw-spot-agent:5002")

# ---- ツール (天気 + ハンドオフ) ----

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

@tool
def handoff_to_spot_agent(query: str) -> str:
    """観光スポットの情報が必要なとき、スポットエージェントに委譲する (A2A ハンドオフ)"""
    client = A2AClient(SPOT_AGENT_URL)
    return client.ask(query)

# ---- トレース収集 ----

def collect_trace(react_agent, query, agent_name, handoff_tool_name, handoff_target, handoff_url):
    """ReAct エージェントを実行し、トレースと最終回答を返す"""
    trace = []
    answer = ""
    pending_calls = {}  # tool_call_id → tool_call

    for event in react_agent.stream({"messages": [("user", query)]}):
        for node_name, node_output in event.items():
            for msg in node_output.get("messages", []):
                # ツール呼び出しの記録
                if node_name == "agent" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        pending_calls[tc["id"]] = tc

                # ツール実行結果の記録
                elif node_name == "tools":
                    call_id = getattr(msg, "tool_call_id", None)
                    tc = pending_calls.pop(call_id, None) if call_id else None
                    if not tc:
                        continue
                    result_text = msg.content if isinstance(msg.content, str) else str(msg.content)

                    if tc["name"] == handoff_tool_name:
                        sub_trace, sub_answer = [], result_text
                        try:
                            parsed = json.loads(result_text)
                            sub_trace = parsed.get("trace", [])
                            sub_answer = parsed.get("answer", result_text)
                        except (json.JSONDecodeError, AttributeError):
                            pass
                        trace.append({
                            "agent": agent_name, "type": "handoff",
                            "target": handoff_target, "url": handoff_url,
                            "query": tc["args"].get("query", ""),
                            "sub_trace": sub_trace, "result": sub_answer,
                        })
                    else:
                        trace.append({
                            "agent": agent_name, "type": "tool",
                            "name": tc["name"], "args": tc["args"],
                            "result": result_text,
                        })

                # 最終回答
                elif node_name == "agent" and msg.content:
                    answer = msg.content

    return trace, answer

# ---- A2A サーバー ----

class WeatherAgentServer(A2AServer):
    def __init__(self):
        card = AgentCard(
            name="Weather Agent (Swarm)",
            description="天気情報を提供し、必要に応じて他のエージェントにハンドオフするエージェント",
            url="http://localhost:5001",
            skills=[
                AgentSkill(name="weather", description="天気情報の検索"),
                AgentSkill(name="handoff", description="観光スポットエージェントへのハンドオフ"),
            ],
        )
        super().__init__(agent_card=card)
        self.react_agent = create_react_agent(
            model=get_llm(),
            tools=[get_weather, calculator, handoff_to_spot_agent],
            prompt=(
                "あなたは天気情報の専門家です。天気に関する質問にはツールを使って回答してください。"
                "観光スポットに関する質問が含まれている場合は、handoff_to_spot_agent を使ってスポットエージェントに委譲してください。"
                "すべての情報が揃ったら、最終的な回答をまとめてください。"
                "日本語で回答してください。"
            ),
        )

    def handle_message(self, message):
        trace, answer = collect_trace(
            self.react_agent, message.content.text,
            agent_name="Weather Agent",
            handoff_tool_name="handoff_to_spot_agent",
            handoff_target="Spot Agent",
            handoff_url=SPOT_AGENT_URL,
        )
        response = json.dumps({"trace": trace, "answer": answer}, ensure_ascii=False)
        return create_text_message(response, role=MessageRole.AGENT)


if __name__ == "__main__":
    print("Weather Agent (Swarm) starting on port 5001...")
    run_server(WeatherAgentServer(), host="0.0.0.0", port=5001)
