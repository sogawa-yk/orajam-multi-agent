FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY .chainlit /app/.chainlit
COPY main.py /app/main.py

EXPOSE 8000

CMD ["chainlit", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
