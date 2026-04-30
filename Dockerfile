# ---- Temel İmaj ----
FROM python:3.11-slim

# ---- Meta Veriler ----
LABEL maintainer="telco-churn-team"
LABEL description="Telco Churn Prediction API"

# ---- Sistem Bağımlılıkları ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# ---- Çalışma Dizini ----
WORKDIR /app

# ---- Bağımlılıkları Kopyala ve Kur ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Proje Dosyalarını Kopyala ----
COPY . .

# ---- Güvenlik: Root Olmayan Kullanıcı ----
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# ---- Uygulama Portu ----
EXPOSE 8000

# ---- Başlatma Komutu ----
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
