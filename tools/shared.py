"""
共通ユーティリティ: OCI Generative AI × LangChain
全ツールカテゴリのデモで共有する
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "openai.gpt-4.1"
BASE_URL = "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com/openai/v1"


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """OCI Generative AI の ChatOpenAI インスタンスを返す"""
    api_key = os.getenv("OCI_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise ValueError("OCI_GENERATIVE_AI_API_KEY is not set. Add it to your .env file.")

    project = os.getenv("OCI_PROJECT")
    if not project:
        raise ValueError("OCI_PROJECT is not set. Add it to your .env file.")

    return ChatOpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        model=MODEL,
        temperature=temperature,
        default_headers={"OpenAI-Project": project},
    )


# ----------------------------------------------------------
# 共通の天気データ (3つのデモで同じデータを返す)
# ----------------------------------------------------------

WEATHER_DATA = {
    "東京": {"city": "東京", "weather": "晴れ", "temperature": 28, "humidity": 55},
    "大阪": {"city": "大阪", "weather": "曇り", "temperature": 24, "humidity": 65},
    "札幌": {"city": "札幌", "weather": "雨",   "temperature": 18, "humidity": 80},
}


# ----------------------------------------------------------
# コンソール出力ヘルパー
# ----------------------------------------------------------

def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_step(label: str, content: str) -> None:
    print(f"\n--- {label} ---")
    print(content)
