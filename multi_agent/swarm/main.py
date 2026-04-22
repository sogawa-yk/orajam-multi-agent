"""
============================================================
スウォーム型マルチエージェント (Swarm)
============================================================
エージェント同士が対等に A2A で通信し、
自律的にタスクをハンドオフ (委譲) し合う。

構成:
  [User] → [Weather Agent] ──A2A handoff──→ [Spot Agent]
                            ←─ 結果を統合して返す ─┘

特徴:
  - 中央の調整役がいない (分散自律型)
  - 各エージェントが自分で判断してハンドオフする
  - エントリーポイントはどのエージェントでもよい
  - スーパーバイザー型との違い:
    スーパーバイザー → 上司が部下に指示を出す
    スウォーム       → 同僚同士が自律的に協力する
============================================================
"""

import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_a2a import A2AClient

# ---- ANSI カラーコード ----

BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
MAGENTA = "\033[35m"
BLUE   = "\033[34m"
WHITE  = "\033[97m"
GRAY   = "\033[90m"
RESET  = "\033[0m"
BG_BLUE   = "\033[44m"
BG_YELLOW = "\033[43m"
BG_GREEN  = "\033[42m"

# ---- 整形出力 ----

def print_header(title: str) -> None:
    print(f"\n{BOLD}{BG_BLUE}{WHITE}  {title}  {RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")


def print_trace(trace: list, indent: int = 0) -> None:
    """トレースを色付きで再帰的に出力する"""
    prefix = "  " * indent
    step = 0

    for item in trace:
        step += 1
        agent = item.get("agent", "?")

        if item["type"] == "tool":
            # ローカルツール実行
            name = item["name"]
            args = item.get("args", {})
            result = item.get("result", "")
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())

            print(f"{prefix}{CYAN}{BOLD}[{agent}]{RESET} {WHITE}ツール実行{RESET}")
            print(f"{prefix}  {GRAY}呼び出し:{RESET}  {name}({args_str})")
            print(f"{prefix}  {GRAY}結果:{RESET}    {GREEN}{result}{RESET}")
            print()

        elif item["type"] == "handoff":
            # A2A ハンドオフ
            target = item.get("target", "?")
            url = item.get("url", "")
            query = item.get("query", "")
            sub_trace = item.get("sub_trace", [])
            result = item.get("result", "")

            print(f"{prefix}{YELLOW}{BOLD}[{agent}]{RESET} {MAGENTA}{BOLD}A2A ハンドオフ → {target}{RESET}")
            print(f"{prefix}  {GRAY}通信先:{RESET}    {DIM}POST {url}/tasks/send{RESET}")
            print(f"{prefix}  {GRAY}依頼内容:{RESET}  {WHITE}\"{query}\"{RESET}")
            print()

            # サブトレース (ハンドオフ先の処理内容)
            if sub_trace:
                print(f"{prefix}  {YELLOW}{'┌' + '─' * 50}{RESET}")
                print(f"{prefix}  {YELLOW}│{RESET} {DIM}{target} の処理:{RESET}")
                print(f"{prefix}  {YELLOW}│{RESET}")
                print_trace(sub_trace, indent=indent + 2)
                print(f"{prefix}  {YELLOW}{'└' + '─' * 50}{RESET}")
                print()


def print_answer(answer: str) -> None:
    print(f"{BOLD}{BG_GREEN}{WHITE}  最終回答  {RESET}")
    print(f"{GREEN}{'─' * 60}{RESET}")
    print(answer)


# ---- エントリーポイント ----
WEATHER_AGENT_URL = os.getenv("WEATHER_AGENT_URL", "http://sw-weather-agent:5001")


def run_swarm(question: str) -> None:
    print_header("スウォーム型マルチエージェント デモ")
    print(f"  {GRAY}質問:{RESET}            {WHITE}{question}{RESET}")
    print(f"  {GRAY}エントリーポイント:{RESET} Weather Agent ({DIM}{WEATHER_AGENT_URL}{RESET})")
    print()

    print_header("A2A 通信トレース")

    client = A2AClient(WEATHER_AGENT_URL)
    raw = client.ask(question)

    # JSON レスポンスをパースして整形出力
    try:
        data = json.loads(raw)
        trace = data.get("trace", [])
        answer = data.get("answer", raw)
    except (json.JSONDecodeError, AttributeError):
        trace = []
        answer = raw

    if trace:
        print_trace(trace)
    else:
        print(f"  {DIM}(トレース情報なし){RESET}\n")

    print()
    print_answer(answer)


if __name__ == "__main__":
    question = "東京の天気と観光スポットを調べて、日帰り旅行プランを提案してください。"
    run_swarm(question)
