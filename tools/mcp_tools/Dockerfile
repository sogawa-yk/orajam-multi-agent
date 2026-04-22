FROM python:3.11-slim

RUN pip install --no-cache-dir \
    langchain-openai \
    langgraph \
    python-dotenv \
    mcp \
    langchain-mcp-adapters

WORKDIR /app
COPY shared.py /app/shared.py
COPY mcp_tools/server.py /app/mcp_tools/server.py
COPY mcp_tools/main.py /app/mcp_tools/main.py

CMD ["python", "mcp_tools/main.py"]
