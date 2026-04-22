FROM python:3.11-slim

RUN pip install --no-cache-dir \
    langchain-openai \
    langgraph \
    python-dotenv

WORKDIR /app
# docker-compose の build.context が orchestration/ なので
# そこからの相対パスでコピーする
COPY shared.py /app/shared.py
COPY react/main.py /app/react/main.py

CMD ["python", "react/main.py"]
