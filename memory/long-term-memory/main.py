import os
from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv
from openai import OpenAI

APP_DIR = Path(__file__).resolve().parent
OCI_BASE_URL = "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com/openai/v1"


def load_root_env() -> None:
    for directory in [APP_DIR, *APP_DIR.parents]:
        env_path = directory / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return


load_root_env()

MODEL = os.getenv("OPENAI_MODEL", "openai.gpt-oss-120b")
MEMORY_SUBJECT_ID = os.getenv("MEMORY_SUBJECT_ID", "demo_user_123456")
MEMORY_WAIT_SECONDS = int(os.getenv("MEMORY_WAIT_SECONDS", "10"))
SYSTEM_PROMPT = """
You are a chat assistant in a long-term memory demo.
Use the memory available for the current memory subject when it is present.
If the user asks what you remember across chats, answer from the available long-term memory.
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
        project = os.getenv("OCI_PROJECT")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    if project:
        client_kwargs["project"] = project

    return OpenAI(**client_kwargs)


def create_conversation_id(client: OpenAI, subject_id: str) -> str:
    conversation = client.conversations.create(
        metadata={"memory_subject_id": subject_id},
    )
    return conversation.id


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Store preference",
            message="I like fish. I don't like shrimp.",
        ),
        cl.Starter(
            label="Recall in new chat",
            message="What do I like?",
        ),
        cl.Starter(
            label="Ask what you remember",
            message="What do you remember about my food preferences?",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory_subject_id", MEMORY_SUBJECT_ID)
    cl.user_session.set("conversation_id", None)

    try:
        client = get_client()
        conversation_id = await cl.make_async(create_conversation_id)(client, MEMORY_SUBJECT_ID)
        cl.user_session.set("conversation_id", conversation_id)
    except Exception as exc:
        await cl.Message(
            content=f"Configuration or API error: {exc}"
        ).send()
        return

    await cl.Message(
        content=(
            "# Long-term memory demo\n\n"
            "This chat creates a new conversation with the same `memory_subject_id` every time you start a new chat.\n"
            f"The current subject is `{MEMORY_SUBJECT_ID}`.\n\n"
            "Try this:\n"
            "1. Say `I like fish. I don't like shrimp.`\n"
            f"2. Wait about {MEMORY_WAIT_SECONDS} seconds for memory processing.\n"
            "3. Click **New Chat**.\n"
            "4. Ask `What do I like?` in the new chat.\n\n"
            "If long-term memory is enabled on the backend for this subject, the assistant should remember across conversations."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    thinking = cl.Message(content="")
    await thinking.send()

    try:
        client = get_client()
        subject_id = cl.user_session.get("memory_subject_id", MEMORY_SUBJECT_ID)
        conversation_id = cl.user_session.get("conversation_id")

        if not conversation_id:
            conversation_id = await cl.make_async(create_conversation_id)(client, subject_id)
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
