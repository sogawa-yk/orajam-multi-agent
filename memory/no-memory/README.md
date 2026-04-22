# No-memory demo

Chainlit front-end + OpenAI Responses API demo that intentionally does not provide memory to the model.

## What this demonstrates

- The chat UI shows a conversation.
- The model does not receive prior turns.
- The assistant can answer only from the latest user message.

Short-term memory and long-term memory are intentionally not implemented.

## Local run

1. Create or update the project root `.env`
2. Set `OPENAI_API_KEY` or `OCI_GENERATIVE_AI_API_KEY`
3. Optionally set `OPENAI_MODEL` such as `openai.gpt-oss-120b`
4. Install dependencies:

```bash
pip install -r memory/no-memory/requirements.txt
```

5. Start the app:

```bash
cd memory/no-memory && chainlit run main.py
```

## Docker run

Build:

```bash
docker build -t no-memory-demo memory/no-memory
```

Run with environment variables:

```bash
docker run --rm -p 8000:8000 --env-file .env no-memory-demo
```

## Implementation note

Each request calls `client.responses.create(...)` with only the latest `message.content`.
No conversation history, `previous_response_id`, short-term memory, or long-term memory store is used.
