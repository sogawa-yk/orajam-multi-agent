FROM python:3.11

RUN ["pip", "install", "openai", "python-dotenv"]

WORKDIR /app
COPY main.py /app/main.py

ENTRYPOINT ["python", "main.py"]
