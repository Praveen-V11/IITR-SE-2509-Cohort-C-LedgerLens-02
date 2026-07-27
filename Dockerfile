FROM python:3.12-slim

# libjpeg/zlib headers for Pillow, curl for the healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libjpeg62-turbo-dev zlib1g-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/ledgerlens

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY streamlit_app.py .

RUN mkdir -p uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1

# Cloud Run passes $PORT; default to 8000 for local/docker-compose runs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
