# Telco Churn Prediction

Müşteri kaybını (churn) tahmin etmek için geliştirilmiş bir makine öğrenmesi projesidir.
**P2P Veri Bilimi Challenge** kapsamında geliştirilmiştir.

Telekomunikasyon sektörüne ait gerçek dünya verisi üzerinde birden fazla model eğitilip karşılaştırılmış; en başarılı model FastAPI aracılığıyla `/predict` endpoint'i olarak sunulmuştur. Gradio tabanlı basit bir web arayüzü de mevcuttur.

---

## İçindekiler

- [Amaç ve Kapsam](#amaç-ve-kapsam)
- [Kullanılan Veri Seti](#kullanılan-veri-seti)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Model Eğitimi ve Karşılaştırma](#model-eğitimi-ve-karşılaştırma)
- [API Kullanımı](#api-kullanımı)
- [Arayüz ile Çalıştırma](#arayüz-ile-çalıştırma)
- [Docker ile Çalıştırma](#docker-ile-çalıştırma)
- [Testler](#testler)
- [Katkı](#katkı)

---

## Amaç ve Kapsam

Bu proje aşağıdaki P2P Challenge gereksinimlerini karşılar:

| Gereksinim | Durum |
|---|---|
| Veri setinin doğru anlaşılması ve işlenmesi | Tamamlandı |
| Modelin doğru eğitilmesi ve çalışması | Tamamlandı (Lojistik Regresyon + Rastgele Orman) |
| Tahmin üretme yeteneği | Tamamlandı (`/predict` endpoint'i) |
| API servisinin düzgün çalışması | Tamamlandı (FastAPI) |
| Kod yapısının anlaşılır ve düzenli olması | Tamamlandı (modüler mimari) |
| Birden fazla model denenmesi ve karşılaştırılması | Tamamlandı (F1 Skoru bazlı seçim) |
| Basit dokümantasyon hazırlanması | Bu dosya |
| Opsiyonel Docker kullanımı | Tamamlandı |
| Opsiyonel basit arayüz geliştirilmesi | Tamamlandı (Gradio) |

---

## Kullanılan Veri Seti

**Kaggle: Telco Customer Churn**
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

- 7.043 müşteri kaydı, 21 özellik
- Hedef değişken: `Churn` (Yes / No)
- Kategorik ve sayısal karışık yapı — one-hot encoding uygulanmıştır
- Keşifsel veri analizi için: `notebooks/eda.ipynb`

---

## Proje Yapısı

```
telco-churn-prediction/
├── api/
│   └── main.py              # FastAPI uygulaması — /predict endpoint'i
├── src/
│   ├── train.py              # Model eğitimi: karşılaştırma ve kaydetme
│   ├── evaluate.py           # Metrik hesaplama ve raporlama
│   ├── preprocess.py         # Veri ön işleme (kodlama, kolon hizalama)
│   └── predict.py            # (Geliştirme aşamasında)
├── ui/
│   └── app.py                # Gradio web arayüzü
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── model.pkl             # En iyi eğitilmiş model (otomatik seçilir)
│   └── feature_columns.pkl  # Eğitimde kullanılan kolon sırası
├── notebooks/
│   └── eda.ipynb             # Keşifsel veri analizi
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

---

## Kurulum

> Gereksinimler: Python 3.11+, pip

**1. Repoyu klonla:**

```bash
git clone https://github.com/<kullanıcı-adı>/telco-churn-prediction.git
cd telco-churn-prediction
```

**2. Sanal ortam oluştur ve aktif et:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**3. Bağımlılıkları yükle:**

```bash
pip install -r requirements.txt
```

**4. Modeli eğit:**

```bash
python src/train.py
```

Çıktı örneği:
```
==================================================
Model 1: Lojistik Regresyon
  Accuracy  : 0.7875
  Precision : 0.6206
  Recall    : 0.5160
  F1 Skoru  : 0.5635
==================================================
Model 2: Rastgele Orman
  Accuracy  : 0.7934
  Precision : 0.6441
  Recall    : 0.5078
  F1 Skoru  : 0.5679
==================================================
Kazanan model: Rastgele Orman (F1: 0.5679)
```

**5. API'yi başlat:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Model Eğitimi ve Karşılaştırma

`src/train.py` çalıştırıldığında:

1. **Lojistik Regresyon** (max_iter=1000) eğitilir.
2. **Rastgele Orman** (n_estimators=100) eğitilir.
3. Her iki modelin **Accuracy / Precision / Recall / F1** metrikleri konsola yazdırılır.
4. **F1 Skoru** en yüksek olan model `models/model.pkl` olarak kaydedilir.

**Neden F1 Skoru?** Churn probleminde veri dengesizdir; yalnızca Accuracy yerine hem Precision hem Recall'ı dengeleyen F1, daha güvenilir bir seçim kriteridir.

**Yeni bir model denemek istersen:**

```python
# src/train.py içinde:
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
```

`compare_models()` fonksiyonuna bu modeli de ekle; kazanan otomatik seçilir.

---

## API Kullanımı

### Sağlık Kontrolü

```bash
curl http://localhost:8000/
```

```json
{"message": "Telco Churn Prediction API is running"}
```

### Churn Tahmini

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": "29.85"
  }'
```

**Yanıt:**

```json
{
  "prediction": 0,
  "churn_probability": 0.4694
}
```

| Alan | Anlamı |
|---|---|
| `prediction` | `0` = Kayıp yok, `1` = Müşteri gidecek (churn) |
| `churn_probability` | Churn olasılığı — 0.0 ile 1.0 arasında |

**İnteraktif Dokümantasyon:** `http://localhost:8000/docs` (FastAPI Swagger UI)

---

## Arayüz ile Çalıştırma

Proje, tarayıcı üzerinden kolayca test edilebilen bir Gradio arayüzüne sahiptir.

**1. API'nin çalıştığından emin ol:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**2. Arayüzü başlat (farklı bir terminalde):**

```bash
python ui/app.py
```

Tarayıcıda otomatik olarak `http://localhost:7860` açılır.

Arayüzde müşteri bilgilerini forma girip **"Tahmin Et"** butonuna tıklaman yeterlidir.

---

## Docker ile Çalıştırma

**Hızlı başlangıç:**

```bash
# 1. Modeli eğit (ilk kurulumda bir kez)
python src/train.py

# 2. Docker ile ayağa kaldır
docker-compose up --build
```

API: `http://localhost:8000`

**Durdurmak için:**

```bash
docker-compose down
```

**Yalnızca Dockerfile ile:**

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

---

## Testler

**API sağlık testi:**

```bash
curl http://localhost:8000/
```

**Model eğitim doğruluğu:**

```bash
python src/train.py
# Çıktı: Her iki modelin metrikleri ve kazananı
```

**Model değerlendirme (ayrı script):**

```bash
python src/evaluate.py
# Kaydedilmiş modeli test verisi üzerinde yeniden raporlar
```

---

## Katkı

1. Repoyu fork'la
2. Yeni branch oluştur: `git checkout -b feature/yeni-model`
3. Değişikliklerini commit et: `git commit -m "feat: XGBoost modeli eklendi"`
4. Push et ve Pull Request aç

---

*P2P Veri Bilimi Challenge — Telco Customer Churn Projesi*
