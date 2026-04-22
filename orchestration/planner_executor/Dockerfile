FROM python:3.11-slim

RUN pip install --no-cache-dir \
    langchain-openai \
    langgraph \
    python-dotenv

WORKDIR /app
COPY shared.py /app/shared.py
COPY planner_executor/main.py /app/planner_executor/main.py

CMD ["python", "planner_executor/main.py"]
