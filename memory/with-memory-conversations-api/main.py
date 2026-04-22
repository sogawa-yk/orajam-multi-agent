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
You are a chat assistant in a with-memory Conversations API demo.
You can use the conversation context carried from earlier turns in the same session.
If the user asks what you remember from the current chat, answer based on the conversation so far.
Keep responses concise and natural.
""".strip()


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OCI_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY or OCI_GENERATIVE_AI_API_KEY is not set. Add one of them to the project root .env."
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


def create_conversation_id(client: OpenAI) -> str:
    conversation = client.conversations.create()
    return conversation.id


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
            label="Ask what you remember",
            message="What do you remember about me from this chat?",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    client = get_client()
    conversation_id = await cl.make_async(create_conversation_id)(client)
    cl.user_session.set("conversation_id", conversation_id)

    await cl.Message(
        content=(
            "# With-memory Conversations API demo\n\n"
            "This chat uses a Conversations API conversation object for the current session.\n"
            "Each turn is sent through the same conversation, so the assistant can carry context across turns.\n\n"
            "Try this:\n"
            "1. Tell the assistant a fact about yourself.\n"
            "2. Ask it to recall that fact in the next turn.\n"
            "3. Start a new chat to create a fresh conversation."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    thinking = cl.Message(content="")
    await thinking.send()

    try:
        client = get_client()
        conversation_id = cl.user_session.get("conversation_id")

        if not conversation_id:
            conversation_id = await cl.make_async(create_conversation_id)(client)
            cl.user_session.set("conversation_id", conversation_id)

        response = await cl.make_async(client.responses.create)(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=message.content,
            conversation=conversation_id,
        )
        thinking.content = response.output_text or "No response text was returned."
    except Exception as exc:
        thinking.content = f"Configuration or API error: {exc}"

    await thinking.update()
