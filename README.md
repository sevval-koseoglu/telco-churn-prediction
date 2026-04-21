# Telco Churn Prediction

Musteri kayip (churn) tahmini yapan bir makine ogrenmesi projesidir.
**P2P Veri Bilimi Challenge** kapsaminda gelistirilmistir.

Telco sektorune ait gercek dunya verisi uzerinde birden fazla model egitilip karsilastirilmis, en basarili model bir FastAPI servisi araciligiyla `/predict` endpoint'i olarak sunulmustur.

---

## Icerik

- [Amac ve Kapsam](#amac-ve-kapsam)
- [Kullanilan Veri Seti](#kullanilan-veri-seti)
- [Proje Yapisi](#proje-yapisi)
- [Kurulum](#kurulum)
- [Model Egitimi ve Karsilastirma](#model-egitimi-ve-karsilastirma)
- [API Kullanimi](#api-kullanimi)
- [Docker ile Calistirma](#docker-ile-calistirma)
- [Testler](#testler)
- [Katki](#katki)

---

## Amac ve Kapsam

Bu proje asagidaki P2P Challenge gereksinimlerini karsilar:

| Gereksinim | Durum |
|---|---|
| Veri setinin dogru anlasilmasi ve islenmesi | Tamamlandi |
| Modelin dogru egitilmesi ve calismasi | Tamamlandi (LogReg + Random Forest) |
| Tahmin uretme yetenegi | Tamamlandi (`/predict` endpoint) |
| API servisinin duzgun calismasi | Tamamlandi (FastAPI) |
| Kod yapisinin anlasilir ve duzenli olmasi | Tamamlandi (modular mimari) |
| Birden fazla model denenmesi ve karsilastirilmasi | Tamamlandi (F1 Score bazli secim) |
| Basit dokumantasyon hazirlanmasi | Bu dosya |
| Opsiyonel Docker kullanimi | Tamamlandi |

---

## Kullanilan Veri Seti

**Kaggle: Telco Customer Churn**
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

- 7043 musteri kaydı, 21 ozellik
- Hedef degisken: `Churn` (Yes / No)
- Kategorik ve sayisal karisik yapi — one-hot encoding uygulanmistir
- EDA icin: `notebooks/eda.ipynb`

---

## Proje Yapisi

```
telco-churn-prediction/
├── api/
│   └── main.py              # FastAPI app — /predict endpoint
├── src/
│   ├── train.py              # Model egitimi: LogReg vs RF, karsilastirma, kaydetme
│   ├── evaluate.py           # Metrik hesaplama ve raporlama
│   ├── preprocess.py         # Veri on-isleme (get_dummies, kolon hizalama)
│   └── predict.py            # (Yedek tahmin modulu — gelistirilecek)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── model.pkl             # En iyi egitilmis model (otomatik secilir)
│   └── feature_columns.pkl  # Egitimde kullanilan kolon siralama listesi
├── notebooks/
│   └── eda.ipynb             # Kesifsel veri analizi
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
git clone https://github.com/<kullanici-adi>/telco-churn-prediction.git
cd telco-churn-prediction
```

**2. Sanal ortam olustur ve aktif et:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**3. Bagimliliklari yukle:**

```bash
pip install -r requirements.txt
```

**4. Modeli egit:**

```bash
python src/train.py
```

Cikti ornegi:
```
==================================================
Model 1: Logistic Regression
  Accuracy  : 0.8027
  Precision : 0.6623
  Recall    : 0.5582
  F1 Score  : 0.6059
==================================================
Model 2: Random Forest
  Accuracy  : 0.7934
  Precision : 0.6441
  Recall    : 0.5078
  F1 Score  : 0.5679
==================================================
Kazanan model: Logistic Regression (F1: 0.6059)
```

**5. API'yi baslat:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Model Egitimi ve Karsilastirma

`src/train.py` calistirildiginda:

1. **Logistic Regression** (max_iter=1000) egitilir.
2. **Random Forest** (n_estimators=100) egitilir.
3. Her iki modelin **Accuracy / Precision / Recall / F1** metrikleri konsola yazdirilir.
4. **F1 Score** en yuksek olan model `models/model.pkl` olarak kaydedilir.

**Neden F1 Score?** Churn probleminde veri dengesizdir; sadece Accuracy yerine hem Precision hem Recall'i dengeleyen F1, daha guvenilir bir secim kriteridir.

**Yeni bir model denemek istiyorsan:**

```python
# src/train.py icinde:
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
```

`compare_models()` fonksiyonuna bu modeli de ekle; kazanan otomatik secer.

---

## API Kullanimi

### Saglik Kontrolu

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

**Cevap:**

```json
{
  "prediction": 0,
  "churn_probability": 0.3742
}
```

| Alan | Anlami |
|---|---|
| `prediction` | `0` = Kayip yok, `1` = Musteri gidecek (churn) |
| `churn_probability` | Churn olasıligi — 0.0 ile 1.0 arasinda |

**Interaktif Dokumantasyon:** `http://localhost:8000/docs` (FastAPI Swagger UI)

---

## Docker ile Calistirma

**Hizli baslangic:**

```bash
# 1. Modeli egit (ilk kurulumda bir kez)
python src/train.py

# 2. Docker ile ayaga kaldir
docker-compose up --build
```

API: `http://localhost:8000`

**Tek komutla durdurmak icin:**

```bash
docker-compose down
```

**Yalnizca Dockerfile ile:**

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

---

## Testler

**API saglik testi:**

```bash
curl http://localhost:8000/
```

**Model egitim dogruluğu:**

```bash
python src/train.py
# Cikti: Her iki modelin metrikleri ve kazanani
```

**Model degerlendirme (ayri script):**

```bash
python src/evaluate.py
# Kaydedilmis modeli test veri seti uzerinde tekrar raporlar
```

---

## Katki

1. Repoyu fork'la
2. Yeni branch olustur: `git checkout -b feature/yeni-model`
3. Degisikliklerini commit et: `git commit -m "feat: XGBoost modeli eklendi"`
4. Push et ve Pull Request ac

---

*P2P Veri Bilimi Challenge — Telco Customer Churn Projesi*
