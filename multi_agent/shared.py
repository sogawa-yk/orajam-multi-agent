"""
共通ユーティリティ: OCI Generative AI × LangChain + デモ用データ
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "openai.gpt-4.1"
BASE_URL = "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com/openai/v1"
PROJECT = "***REMOVED***"


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.getenv("OCI_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise ValueError("OCI_GENERATIVE_AI_API_KEY is not set.")
    return ChatOpenAI(
        base_url=BASE_URL, api_key=api_key, model=MODEL,
        temperature=temperature,
        default_headers={"OpenAI-Project": PROJECT},
    )


# ----------------------------------------------------------
# デモ用データ (全エージェントで共通)
# ----------------------------------------------------------

WEATHER_DATA = {
    "東京": {"city": "東京", "weather": "晴れ", "temperature": 28, "humidity": 55},
    "大阪": {"city": "大阪", "weather": "曇り", "temperature": 24, "humidity": 65},
    "札幌": {"city": "札幌", "weather": "雨",   "temperature": 18, "humidity": 80},
}

SPOT_DATA = {
    "東京": [
        {"name": "浅草寺", "description": "東京最古の寺院。雷門が有名", "category": "歴史"},
        {"name": "東京スカイツリー", "description": "高さ634mの電波塔。展望台あり", "category": "観光"},
        {"name": "渋谷スクランブル交差点", "description": "世界最大級のスクランブル交差点", "category": "体験"},
    ],
    "大阪": [
        {"name": "大阪城", "description": "豊臣秀吉が築いた名城", "category": "歴史"},
        {"name": "道頓堀", "description": "グリコ看板で有名な繁華街", "category": "グルメ"},
    ],
    "札幌": [
        {"name": "大通公園", "description": "札幌の中心に広がる都市公園", "category": "自然"},
        {"name": "札幌時計台", "description": "札幌のシンボル的建造物", "category": "歴史"},
    ],
}


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
