FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api
COPY ingest/ ./ingest
COPY data/ ./data

EXPOSE 8000

CMD ["uvicorn", "api.search_service:app", "--host", "0.0.0.0", "--port", "8000"]