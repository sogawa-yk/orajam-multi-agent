"""
============================================================
Planner-Executor パターン (計画と実行の分離) — LangGraph 実装
============================================================
「計画を立てるエージェント」と「実行するエージェント」を分離するパターン。

グラフ構造:
  [START] → plan → execute → replan → should_continue? ─┐
                      ↑                                   │
                      └──── (追加ステップあり) ────────────┘
                                       │
                               (計画完了)
                                       ↓
                                     [END]

  - plan ノード    : タスクを分析し、ステップのリストを作成する
  - execute ノード : 未実行のステップを1つ実行する
  - replan ノード  : 実行結果を見て、計画の修正が必要か判断する
  - 条件分岐      : 未実行ステップがあれば execute へ、なければ END へ

参考: Wang et al., "Plan-and-Solve Prompting" (2023)
============================================================
"""

import json
import re
import sys
import os
from typing import TypedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from shared import get_llm, print_header, print_step

# ----------------------------------------------------------
# State
# ----------------------------------------------------------

class Step(TypedDict):
    id: int
    task: str
    result: str  # 空文字なら未実行


class PlannerState(TypedDict):
    task: str               # ユーザーからのタスク
    goal: str               # 計画の目標
    steps: list[Step]       # ステップ一覧 (結果込み)
    current_step_idx: int   # 次に実行するステップのインデックス
    final_answer: str       # 最終まとめ


# ----------------------------------------------------------
# プロンプト
# ----------------------------------------------------------

PLANNER_PROMPT = """あなたは計画立案の専門家です。
ユーザーのタスクを分析し、実行可能なステップに分解してください。

以下のJSON形式で計画を出力してください:
{
  "goal": "最終的な目標",
  "steps": [
    {"id": 1, "task": "具体的なステップの説明"},
    {"id": 2, "task": "具体的なステップの説明"}
  ]
}

ルール:
- 各ステップは具体的かつ実行可能にする
- 3〜5ステップ程度に収める
- JSON以外は出力しない"""

EXECUTOR_PROMPT = """あなたは実行担当のアシスタントです。
与えられたステップを実行し、結果を簡潔に報告してください。
これまでの実行結果も参考にして回答してください。"""

REPLAN_PROMPT = """あなたは計画立案の専門家です。
元のタスクと実行結果を確認し、追加で必要なステップがあるか判断してください。

追加ステップがある場合は以下のJSON形式で出力してください:
{
  "steps": [
    {"id": 10, "task": "追加ステップの説明"}
  ]
}

追加不要なら以下を出力してください:
{"steps": []}"""

# ----------------------------------------------------------
# ヘルパー
# ----------------------------------------------------------

llm = get_llm(temperature=0.2)


def parse_json(text: str) -> dict | None:
    """LLMの出力からJSONを抽出する"""
    try:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return None


def format_results(steps: list[Step]) -> str:
    """実行済みステップの結果を整形する"""
    done = [s for s in steps if s["result"]]
    if not done:
        return "(まだ実行結果はありません)"
    return "\n\n".join(
        f"--- ステップ {s['id']}: {s['task']} ---\n{s['result']}" for s in done
    )


def print_plan(goal: str, steps: list[Step]) -> None:
    """計画をコンソールに見やすく表示する"""
    print(f"\n  目標: {goal}")
    print(f"  ステップ数: {len(steps)}")
    print()
    for s in steps:
        status = "✓" if s["result"] else " "
        print(f"    [{status}] {s['id']}. {s['task']}")


# ----------------------------------------------------------
# ノード関数
# ----------------------------------------------------------

def plan(state: PlannerState) -> PlannerState:
    """計画を立案するノード"""
    print_step("Phase 1: 計画立案 (Planner)", "タスクを分析しステップに分解中...")

    response = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=state["task"]),
    ])

    parsed = parse_json(response.content)
    if not parsed:
        print("  !! 計画のパースに失敗しました")
        return state

    goal = parsed.get("goal", state["task"])
    steps = [
        Step(id=s["id"], task=s["task"], result="")
        for s in parsed.get("steps", [])
    ]

    print_plan(goal, steps)

    return {"goal": goal, "steps": steps, "current_step_idx": 0}


def execute(state: PlannerState) -> PlannerState:
    """現在のステップを実行するノード"""
    idx = state["current_step_idx"]
    steps = list(state["steps"])  # コピー
    step = steps[idx]

    print_step(
        f"Phase 2: ステップ {step['id']} を実行 (Executor)",
        f"タスク: {step['task']}",
    )

    context = format_results(steps)

    response = llm.invoke([
        SystemMessage(content=EXECUTOR_PROMPT),
        HumanMessage(content=(
            f"【全体の目標】\n{state['goal']}\n\n"
            f"【これまでの結果】\n{context}\n\n"
            f"【今回のステップ】\n{step['task']}\n\n"
            f"上記のステップを実行してください。"
        )),
    ])

    result = response.content
    print(result)

    # 結果を保存してインデックスを進める
    steps[idx] = Step(id=step["id"], task=step["task"], result=result)
    return {"steps": steps, "current_step_idx": idx + 1}


def replan(state: PlannerState) -> PlannerState:
    """計画を見直すノード"""
    print_step("Phase 3: 計画の見直し (Replan)", "追加ステップが必要か判断中...")

    response = llm.invoke([
        SystemMessage(content=REPLAN_PROMPT),
        HumanMessage(content=(
            f"【元のタスク】\n{state['task']}\n\n"
            f"【実行結果】\n{format_results(state['steps'])}\n\n"
            f"追加ステップは必要ですか？"
        )),
    ])

    parsed = parse_json(response.content)
    steps = list(state["steps"])

    if parsed and parsed.get("steps"):
        new_steps = [
            Step(id=s["id"], task=s["task"], result="")
            for s in parsed["steps"]
        ]
        steps.extend(new_steps)
        print(f"  >> {len(new_steps)} 件の追加ステップあり")
        print_plan(state["goal"], steps)
    else:
        print("  >> 計画の修正は不要です")

    return {"steps": steps}


def summarize(state: PlannerState) -> PlannerState:
    """全実行結果をまとめるノード"""
    print_step("最終まとめ", "全ステップの結果を統合中...")

    response = llm.invoke([
        SystemMessage(content="すべての実行結果をまとめて、最終的な回答を作成してください。"),
        HumanMessage(content=(
            f"【タスク】\n{state['task']}\n\n"
            f"【全ステップの実行結果】\n{format_results(state['steps'])}"
        )),
    ])

    return {"final_answer": response.content}


# ----------------------------------------------------------
# 条件分岐
# ----------------------------------------------------------

def should_execute_more(state: PlannerState) -> str:
    """未実行のステップがあれば execute へ、なければ replan へ"""
    if state["current_step_idx"] < len(state["steps"]):
        return "execute"
    return "replan"


def should_continue_after_replan(state: PlannerState) -> str:
    """replan で追加ステップがあれば execute へ、なければ summarize へ"""
    # current_step_idx がまだ steps の範囲内なら追加ステップがある
    if state["current_step_idx"] < len(state["steps"]):
        return "execute"
    return "summarize"


# ----------------------------------------------------------
# グラフ構築
# ----------------------------------------------------------

def build_graph():
    """Planner-Executor パターンの StateGraph を構築する"""
    graph = StateGraph(PlannerState)

    # ノード追加
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("replan", replan)
    graph.add_node("summarize", summarize)

    # エッジ
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        should_execute_more,
        {"execute": "execute", "replan": "replan"},
    )
    graph.add_conditional_edges(
        "execute",
        should_execute_more,
        {"execute": "execute", "replan": "replan"},
    )
    graph.add_conditional_edges(
        "replan",
        should_continue_after_replan,
        {"execute": "execute", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)

    return graph.compile()


# ----------------------------------------------------------
# 実行
# ----------------------------------------------------------

if __name__ == "__main__":
    task = "東京タワーと富士山の高さをそれぞれ調べて、2つの高さの差を計算し、わかりやすく説明してください。"

    print_header("Planner-Executor パターン デモ (LangGraph)")
    print(f"\nタスク: {task}")

    app = build_graph()

    result = app.invoke({
        "task": task,
        "goal": "",
        "steps": [],
        "current_step_idx": 0,
        "final_answer": "",
    })

    print_header("最終結果")
    print(result["final_answer"])
