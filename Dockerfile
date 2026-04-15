FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8501 8000

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]
