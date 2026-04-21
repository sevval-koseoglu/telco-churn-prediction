# ---- Base Image ----
FROM python:3.11-slim

# ---- Metadata ----
LABEL maintainer="telco-churn-team"
LABEL description="Telco Churn Prediction API"

# ---- Sistem bagimliliklar ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# ---- Calisma dizini ----
WORKDIR /app

# ---- Bagimliliklari kopyala ve yukle ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Proje dosyalarini kopyala ----
COPY . .

# ---- Guvenlik: root olmayan kullanici ----
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# ---- Port ----
EXPOSE 8000

# ---- Baslat ----
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
