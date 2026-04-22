# With-memory Responses API demo

Chainlit front-end + OpenAI Responses API demo that preserves conversation state within the current chat session.

## What this demonstrates

- The chat UI shows a conversation.
- The model can retain context across turns in the current session.
- The assistant can recall facts mentioned earlier in the same chat.

This demo uses session-level memory only. It does not implement long-term memory or any external memory store.

## Local run

1. Create or update the project root `.env`
2. Set `OPENAI_API_KEY` or `OCI_GENERATIVE_AI_API_KEY`
3. Optionally set `OPENAI_MODEL` such as `openai.gpt-oss-120b`
4. Install dependencies:

```bash
pip install -r memory/with-memory-responses-api/requirements.txt
```

5. Start the app:

```bash
cd memory/with-memory-responses-api && chainlit run main.py
```

## Docker run

Build:

```bash
docker build -t with-memory-demo memory/with-memory-responses-api
```

Run with environment variables:

```bash
docker run --rm -p 8001:8000 --env-file .env with-memory-demo
```

## Implementation note

Each request calls `client.responses.create(...)` and passes the prior `response.id` as `previous_response_id`.
That keeps conversation state across turns in the same session without introducing long-term memory.
