"""
共通ユーティリティ: OCI Generative AI × LangChain
全パターンで共有するLLMインスタンスを提供する
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "openai.gpt-4.1"
BASE_URL = "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com/openai/v1"
PROJECT = "***REMOVED***"


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """OCI Generative AI の ChatOpenAI インスタンスを返す"""
    api_key = os.getenv("OCI_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise ValueError("OCI_GENERATIVE_AI_API_KEY is not set. Add it to your .env file.")

    return ChatOpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        model=MODEL,
        temperature=temperature,
        default_headers={"OpenAI-Project": PROJECT},
    )


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
