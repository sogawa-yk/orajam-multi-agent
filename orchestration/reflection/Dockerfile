FROM python:3.11-slim

RUN pip install --no-cache-dir \
    langchain-openai \
    langgraph \
    python-dotenv

WORKDIR /app
COPY shared.py /app/shared.py
COPY reflection/main.py /app/reflection/main.py

CMD ["python", "reflection/main.py"]
