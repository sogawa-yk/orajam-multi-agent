# Long-term memory demo

Chainlit front-end demo for cross-conversation memory using conversation metadata.

## What this demonstrates

- Each chat session starts a new conversation.
- Every conversation is created with the same `memory_subject_id`.
- If the backend supports long-term memory for that subject, the model can recall facts across separate conversations.

## Local run

1. Create or update the project root `.env`
2. Set `OPENAI_API_KEY` or `OCI_GENERATIVE_AI_API_KEY`
3. Optionally set `OPENAI_MODEL`, `MEMORY_SUBJECT_ID`, and `MEMORY_WAIT_SECONDS`
4. Install dependencies:

```bash
pip install -r memory/long-term-memory/requirements.txt
```

5. Start the app:

```bash
cd memory/long-term-memory && chainlit run main.py
```

## Docker run

Build:

```bash
docker build -t long-term-memory-demo memory/long-term-memory
```

Run with environment variables:

```bash
docker run --rm -p 8003:8000 --env-file .env long-term-memory-demo
```

## Demo flow

1. In the first chat, say `I like fish. I don't like shrimp.`
2. Wait about `MEMORY_WAIT_SECONDS` seconds.
3. Start a new chat.
4. Ask `What do I like?`

## Implementation note

The app creates a new conversation for each Chainlit chat with:

```python
client.conversations.create(
    metadata={"memory_subject_id": subject_id},
)
```

Each turn then calls `client.responses.create(...)` with that conversation ID.
