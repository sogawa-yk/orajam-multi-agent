FROM python:3.11-slim

RUN pip install --no-cache-dir \
    langchain-openai \
    langgraph \
    python-dotenv \
    python-a2a

WORKDIR /app
COPY shared.py /app/shared.py
COPY supervisor/ /app/supervisor/
