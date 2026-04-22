FROM python:3.11-slim

RUN pip install --no-cache-dir \
    langchain-openai \
    langgraph \
    python-dotenv

WORKDIR /app
COPY shared.py /app/shared.py
COPY local_tools/main.py /app/local_tools/main.py

CMD ["python", "local_tools/main.py"]
