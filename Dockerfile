FROM python:3.9-slim

ENV PYTHONUTF8=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

WORKDIR /app/src

CMD ["python", "agent.py"]