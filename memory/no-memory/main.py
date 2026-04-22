import os
from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv
from openai import OpenAI

APP_DIR = Path(__file__).resolve().parent
OCI_BASE_URL = "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com/openai/v1"
OCI_PROJECT = "***REMOVED***"


def load_root_env() -> None:
    for directory in [APP_DIR, *APP_DIR.parents]:
        env_path = directory / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return


load_root_env()

MODEL = os.getenv("OPENAI_MODEL", "openai.gpt-oss-120b")
SYSTEM_PROMPT = """
You are a chat assistant in a no-memory demo.
You only receive the user's latest message for each request.
Do not claim to remember any earlier turn unless the user repeats that information in the current message.
If the user asks you to recall previous context, explain that it was not included in the current request.
Keep responses concise and natural.
""".strip()


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OCI_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY or OCI_GENERATIVE_AI_API_KEY is not set. Add one of them to the environment."
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    project = os.getenv("OPENAI_PROJECT")

    if os.getenv("OCI_GENERATIVE_AI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        base_url = os.getenv("OCI_BASE_URL", OCI_BASE_URL)
        project = os.getenv("OCI_PROJECT", OCI_PROJECT)

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    if project:
        client_kwargs["project"] = project

    return OpenAI(**client_kwargs)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Remember my favorite food",
            message="My favorite food is ramen. Please remember it for later.",
        ),
        cl.Starter(
            label="Check whether you remember",
            message="What is my favorite food?",
        ),
        cl.Starter(
            label="Force the context into one turn",
            message="Earlier I said my favorite food is ramen. What is my favorite food?",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "# No-memory demo\n\n"
            "This chat intentionally does not send conversation history to the model.\n"
            "The UI shows previous messages, but the model only receives your latest message each time.\n\n"
            "Try this:\n"
            "1. Tell the assistant a fact about yourself.\n"
            "2. Ask it to recall that fact in the next turn.\n"
            "3. Repeat the fact inside the same message and compare the result."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    thinking = cl.Message(content="")
    await thinking.send()

    try:
        response = await cl.make_async(get_client().responses.create)(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=message.content,
        )
        thinking.content = response.output_text or "No response text was returned."
    except Exception as exc:
        thinking.content = f"Configuration or API error: {exc}"
    await thinking.update()
