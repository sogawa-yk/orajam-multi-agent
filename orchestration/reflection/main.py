"""
============================================================
Reflection パターン (自己反省) — LangGraph 実装
============================================================
LLMが自身の出力を批評し、改善を繰り返すパターン。

グラフ構造:
  [START] → generate → reflect → should_continue? ─┐
                ↑                                    │
                └──── (スコア < 閾値なら) ───────────┘
                                    │
                            (スコア >= 閾値なら)
                                    ↓
                                  [END]

  - generate ノード : ライターが文章を生成/改善する
  - reflect ノード  : 批評家が文章を評価しスコアをつける
  - 条件分岐       : スコアが閾値以上なら END、未満なら generate に戻る

参考: Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
============================================================
"""

import re
import sys
import os
from typing import TypedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from shared import get_llm, print_header, print_step

# ----------------------------------------------------------
# State (グラフ全体で共有するデータ)
# ----------------------------------------------------------

class ReflectionState(TypedDict):
    task: str           # ユーザーからのタスク
    draft: str          # 現在の文章
    reflection: str     # 批評内容
    score: int          # 批評スコア (0-10)
    iteration: int      # 現在のイテレーション


# ----------------------------------------------------------
# プロンプト
# ----------------------------------------------------------

GENERATOR_PROMPT = """あなたは優秀なライターです。
ユーザーの指示に従って、わかりやすく正確な文章を書いてください。"""

REFLECTOR_PROMPT = """あなたは厳格な批評家です。
与えられた文章を以下の観点で評価し、具体的な改善点を指摘してください:

1. 正確性: 事実に誤りがないか
2. 網羅性: 重要な情報が欠けていないか
3. 明瞭性: わかりやすく書かれているか
4. 構成: 論理的な流れになっているか

最後に「スコア: X/10」の形式で総合スコアをつけてください。
8点以上なら十分な品質です。"""

IMPROVER_PROMPT = """あなたは優秀なライターです。
元の文章と批評を踏まえて、改善版の文章を書いてください。
批評で指摘された点をすべて反映し、文章全体を書き直してください。"""

# ----------------------------------------------------------
# ノード関数 (グラフの各ノードの処理)
# ----------------------------------------------------------

llm = get_llm()


def generate(state: ReflectionState) -> ReflectionState:
    """文章を生成/改善するノード"""
    iteration = state["iteration"]

    if iteration == 0:
        # 初回: タスクから直接生成
        print_step(f"生成 (初回)", "ライターが文章を作成中...")
        response = llm.invoke([
            SystemMessage(content=GENERATOR_PROMPT),
            HumanMessage(content=state["task"]),
        ])
    else:
        # 2回目以降: 批評をもとに改善
        print_step(f"改善 (イテレーション {iteration})", "ライターが批評をもとに改善中...")
        response = llm.invoke([
            SystemMessage(content=IMPROVER_PROMPT),
            HumanMessage(content=(
                f"【元の文章】\n{state['draft']}\n\n"
                f"【批評】\n{state['reflection']}\n\n"
                f"上記の批評を踏まえて、改善版を書いてください。"
            )),
        ])

    draft = response.content
    print(draft)
    return {"draft": draft, "iteration": iteration + 1}


def reflect(state: ReflectionState) -> ReflectionState:
    """文章を批評するノード"""
    print_step(f"批評 (イテレーション {state['iteration']})", "批評家が文章を評価中...")

    response = llm.invoke([
        SystemMessage(content=REFLECTOR_PROMPT),
        HumanMessage(content=f"以下の文章を批評してください:\n\n{state['draft']}"),
    ])

    reflection = response.content
    score = extract_score(reflection)

    print(reflection)
    print(f"\n  >> スコア: {score}/10")

    return {"reflection": reflection, "score": score}


def extract_score(text: str) -> int:
    """批評文からスコアを抽出する"""
    match = re.search(r"スコア\s*[:：]\s*(\d+)\s*/\s*10", text)
    return int(match.group(1)) if match else 0


# ----------------------------------------------------------
# 条件分岐 (ループ継続の判定)
# ----------------------------------------------------------

SCORE_THRESHOLD = 8
MAX_ITERATIONS = 3


def should_continue(state: ReflectionState) -> str:
    """スコアが閾値以上、または最大回数に達したら終了"""
    if state["score"] >= SCORE_THRESHOLD:
        print(f"\n  >> 品質基準を満たしました (>= {SCORE_THRESHOLD}/10)")
        return "end"
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"\n  >> 最大イテレーション数 ({MAX_ITERATIONS}) に到達")
        return "end"
    print(f"\n  >> 基準未達 (< {SCORE_THRESHOLD}/10) → 改善ループへ")
    return "continue"


# ----------------------------------------------------------
# グラフ構築
# ----------------------------------------------------------

def build_graph():
    """Reflection パターンの StateGraph を構築する"""
    graph = StateGraph(ReflectionState)

    # ノード追加
    graph.add_node("generate", generate)
    graph.add_node("reflect", reflect)

    # エッジ: START → generate → reflect → 条件分岐
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges(
        "reflect",
        should_continue,
        {"continue": "generate", "end": END},
    )

    return graph.compile()


# ----------------------------------------------------------
# 実行
# ----------------------------------------------------------

if __name__ == "__main__":
    task = "東京タワーと富士山の高さをそれぞれ調べて、2つの高さの差を計算し、わかりやすく説明してください。"

    print_header("Reflection パターン デモ (LangGraph)")
    print(f"\nタスク: {task}")

    app = build_graph()

    # グラフ実行
    result = app.invoke({
        "task": task,
        "draft": "",
        "reflection": "",
        "score": 0,
        "iteration": 0,
    })

    print_header("最終結果")
    print(result["draft"])
