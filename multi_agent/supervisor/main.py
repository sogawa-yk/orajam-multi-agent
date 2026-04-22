"""
============================================================
スーパーバイザー型マルチエージェント (Supervisor)
============================================================
中央のスーパーバイザーがタスクを受け取り、
ワーカーエージェントに A2A で委譲する。

構成:
  [User] → [Supervisor (ReAct)] ──A2A──→ [Weather Agent]
                                 ──A2A──→ [Spot Agent]

特徴:
  - スーパーバイザーが全体を制御 (中央集権)
  - ワーカーは独立した A2A サーバー
  - スーパーバイザーの「ツール」= A2A クライアント呼び出し
============================================================
"""

import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from python_a2a import A2AClient
from shared import get_llm

# ---- ANSI カラーコード ----

BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
YELLOW  = "\033[33m"
GREEN   = "\033[32m"
MAGENTA = "\033[35m"
BLUE    = "\033[34m"
WHITE   = "\033[97m"
GRAY    = "\033[90m"
RESET   = "\033[0m"
BG_BLUE  = "\033[44m"
BG_GREEN = "\033[42m"

# ---- ワーカーエージェントの URL ----

WEATHER_AGENT_URL = os.getenv("WEATHER_AGENT_URL", "http://sv-weather-agent:5001")
SPOT_AGENT_URL    = os.getenv("SPOT_AGENT_URL",    "http://sv-spot-agent:5002")

WORKER_META = {
    "ask_weather_agent": ("Weather Agent", WEATHER_AGENT_URL),
    "ask_spot_agent":    ("Spot Agent",    SPOT_AGENT_URL),
}

# ---- スーパーバイザーのツール = A2A でワーカーを呼び出す ----

@tool
def ask_weather_agent(query: str) -> str:
    """天気エージェントに質問する。天気・気温・湿度について聞きたいときに使う。"""
    client = A2AClient(WEATHER_AGENT_URL)
    return client.ask(query)

@tool
def ask_spot_agent(query: str) -> str:
    """観光スポットエージェントに質問する。観光地・名所について聞きたいときに使う。"""
    client = A2AClient(SPOT_AGENT_URL)
    return client.ask(query)

# ---- 整形出力 ----

def print_header(title: str) -> None:
    print(f"\n{BOLD}{BG_BLUE}{WHITE}  {title}  {RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")


def print_sub_trace(trace: list, indent: int = 2) -> None:
    """ワーカーのサブトレースを罫線内に表示する"""
    prefix = "  " * indent
    for item in trace:
        agent = item.get("agent", "?")
        name  = item.get("name", "")
        args  = item.get("args", {})
        result = item.get("result", "")
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())

        print(f"{prefix}{CYAN}{BOLD}[{agent}]{RESET} {WHITE}ツール実行{RESET}")
        print(f"{prefix}  {GRAY}呼び出し:{RESET}  {name}({args_str})")
        print(f"{prefix}  {GRAY}結果:{RESET}    {GREEN}{result}{RESET}")
        print()


def print_answer(answer: str) -> None:
    print(f"\n{BOLD}{BG_GREEN}{WHITE}  最終回答  {RESET}")
    print(f"{GREEN}{'─' * 60}{RESET}")
    print(answer)


# ---- スーパーバイザー実行 ----

def run_supervisor(question: str) -> None:
    llm = get_llm()

    supervisor = create_react_agent(
        model=llm,
        tools=[ask_weather_agent, ask_spot_agent],
        prompt=(
            "あなたは旅行プランナーのスーパーバイザーです。"
            "ユーザーの質問に答えるために、天気エージェントと観光スポットエージェントに適切に仕事を委譲してください。"
            "各エージェントの回答をもとに、最終的な旅行プランをまとめてください。"
            "日本語で回答してください。"
        ),
    )

    print_header("A2A 通信トレース")

    answer = ""
    pending = {}

    for event in supervisor.stream({"messages": [("user", question)]}):
        for node_name, node_output in event.items():
            for msg in node_output.get("messages", []):
                # Supervisor がツール (= A2A 委譲) を呼び出す
                if node_name == "agent" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        pending[tc["id"]] = tc
                        tool_name = tc["name"]
                        target, url = WORKER_META.get(tool_name, (tool_name, ""))
                        query = tc["args"].get("query", "")

                        print(f"{YELLOW}{BOLD}[Supervisor]{RESET} {MAGENTA}{BOLD}A2A 委譲 → {target}{RESET}")
                        print(f"  {GRAY}通信先:{RESET}    {DIM}POST {url}/tasks/send{RESET}")
                        print(f"  {GRAY}依頼内容:{RESET}  {WHITE}\"{query}\"{RESET}")
                        print()

                # ワーカーからの応答
                elif node_name == "tools":
                    call_id = getattr(msg, "tool_call_id", None)
                    tc = pending.pop(call_id, None)
                    if not tc:
                        continue

                    tool_name = tc["name"]
                    target, _ = WORKER_META.get(tool_name, (tool_name, ""))
                    raw = msg.content if isinstance(msg.content, str) else str(msg.content)

                    # ワーカーの JSON レスポンスをパースしてサブトレースを表示
                    sub_trace = []
                    try:
                        parsed = json.loads(raw)
                        sub_trace = parsed.get("trace", [])
                    except (json.JSONDecodeError, AttributeError):
                        pass

                    if sub_trace:
                        print(f"  {YELLOW}{'┌' + '─' * 50}{RESET}")
                        print(f"  {YELLOW}│{RESET} {DIM}{target} の処理:{RESET}")
                        print(f"  {YELLOW}│{RESET}")
                        print_sub_trace(sub_trace)
                        print(f"  {YELLOW}{'└' + '─' * 50}{RESET}")
                        print()

                # Supervisor の最終回答
                elif node_name == "agent" and msg.content:
                    answer = msg.content

    print_answer(answer)


if __name__ == "__main__":
    question = "東京の天気と観光スポットを調べて、日帰り旅行プランを提案してください。"

    print_header("スーパーバイザー型マルチエージェント デモ")
    print(f"  {GRAY}質問:{RESET}  {WHITE}{question}{RESET}")
    print()

    run_supervisor(question)
