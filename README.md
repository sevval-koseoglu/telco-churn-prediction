# Telco Churn Prediction

Musteri kayip (churn) tahmini yapan bir makine ogrenmesi projesidir. Veri Bilimi kapsaminda gelistirilmis olup, FastAPI tabanli bir REST API sunmaktadir. Proje, mevcut Logistic Regression modeli ile tahmin yapar ve ikinci bir model denemesine hazir bir altyapi sunar.

---

## Icerik

- [Amac](#amac)
- [Proje Yapisi](#proje-yapisi)
- [Kurulum](#kurulum)
- [API Kullanimi](#api-kullanimi)
- [Model Denemeleri](#model-denemeleri)
- [Docker ile Calistirma](#docker-ile-calistirma)
- [Testler](#testler)
- [Katki](#katki)
- [Lisans](#lisans)

---

## Amac

Bu proje, bir telekomunikasyon sirketinin musteri verilerini kullanarak musterilerin kayip (churn) durumunu tahmin etmeyi amaclar. Temel hedefler:

- **Tahmin API'si:** `/predict` endpoint'i ile musteri verisi alip churn tahmini dondurme.
- **Model Karsilastirma:** Farkli makine ogrenmesi modellerini deneyerek en iyi performansi bulma.
- **Docker Dagitimi:** Projeyi konteyner ortaminda calistirarak tekrarlanabilir dagitim saglama.

---

## Proje Yapisi

```
telco-churn-prediction/
├── api/
│   └── main.py              # FastAPI uygulamasi ve /predict endpoint'i
├── src/
│   ├── train.py              # Model egitim scripti
│   ├── predict.py            # Tahmin yardimci modulu (gelistirilecek)
│   ├── preprocess.py         # Veri on-isleme modulu (gelistirilecek)
│   └── evaluate.py           # Model degerlendirme modulu (gelistirilecek)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Ham veri seti
├── models/                   # Egitilmis model dosyalari (.pkl)
├── notebooks/
│   └── eda.ipynb             # Kesifsel veri analizi notebook'u
├── config.py                 # Konfiguerasyon ayarlari
├── requirements.txt          # Python bagimliliklari
├── Dockerfile                # Docker imaj tanimlamasi
├── docker-compose.yml        # Docker Compose konfiguerasyonu
├── .dockerignore             # Docker build'den haric tutulan dosyalar
└── .gitignore                # Git'ten haric tutulan dosyalar
```

---

## Kurulum

### Gereksinimler

- Python 3.11 veya ustu
- pip (Python paket yoneticisi)

### Adimlar

1. **Repoyu klonlayin:**

```bash
git clone https://github.com/<kullanici-adi>/telco-churn-prediction.git
cd telco-churn-prediction
```

2. **Sanal ortam olusturun ve aktif edin:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

3. **Bagimliliklari yukleyin:**

```bash
pip install -r requirements.txt
```

4. **Models klasorunu olusturun (yoksa):**

```bash
mkdir models
```

5. **Modeli egitin:**

```bash
python src/train.py
```

Bu komut `models/model.pkl` ve `models/feature_columns.pkl` dosyalarini olusturur.

6. **API'yi baslatin:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API suan `http://localhost:8000` adresinde calismaktadir.

---

## API Kullanimi

### Saglik Kontrolu

```
GET /
```

**Cevap:**
```json
{
  "message": "Telco Churn Prediction API is running"
}
```

### Churn Tahmini

```
POST /predict
Content-Type: application/json
```

**Ornek Istek (Request Body):**

```json
{
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
}
```

**Ornek Cevap:**

```json
{
  "prediction": 0,
  "churn_probability": 0.3742
}
```

| Alan | Aciklama |
|------|----------|
| `prediction` | 0 = Kayip yok, 1 = Kayip (churn) |
| `churn_probability` | Churn olma olasiligi (0.0 - 1.0) |

**curl ile test ornegi:**

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

---

## Model Denemeleri

Projede su anda **Logistic Regression** kullanilmaktadir. Ikinci bir model denemek icin asagidaki adimlari izleyin.

### Adim 1: Yeni Model Secimi

Onerilern alternatif modeller:

| Model | Import | Avantaj |
|-------|--------|---------|
| Random Forest | `from sklearn.ensemble import RandomForestClassifier` | Non-linear iliskiler, feature importance |
| Gradient Boosting | `from sklearn.ensemble import GradientBoostingClassifier` | Yuksek dogruluk |
| XGBoost | `from xgboost import XGBClassifier` | Hiz ve performans |

### Adim 2: train.py Dosyasini Degistirme

`src/train.py` dosyasinda model tanimini degistirin:

```python
# ONCEKI (Logistic Regression):
# model = LogisticRegression(max_iter=1000)

# YENI (Random Forest ornegi):
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

Egitim sonrasi farkli dosya adlariyla kaydetmek isterseniz:

```python
with open("models/model_rf.pkl", "wb") as f:
    pickle.dump(model, f)
```

### Adim 3: API'yi Yeni Modelle Guncelleme

`api/main.py` dosyasinda model dosya yolunu degistirin:

```python
with open("models/model_rf.pkl", "rb") as f:
    model = pickle.load(f)
```

### Adim 4: Karsilastirma ve Degerlendirme

Her iki modeli karsilastirmak icin `src/evaluate.py` dosyasina asagidaki gibi metrikler ekleyin:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Adim 5: Yeni Bagimliliklari Ekleme

Eger XGBoost gibi ek kutuphane gerekiyorsa `requirements.txt` dosyasina ekleyin:

```
xgboost
```

Ardindan Docker imajini yeniden build edin:

```bash
docker-compose build
docker-compose up
```

---

## Docker ile Calistirma

### Onkosuller

- Docker ve Docker Compose yuklu olmalidir.

### Hizli Baslangic

1. **Modeli egitin (henuz yapilmadiysa):**

```bash
python src/train.py
```

2. **Docker imajini build edin ve calistirin:**

```bash
docker-compose up --build
```

3. **API'ye erisin:**

```
http://localhost:8000
```

### Yalnizca Dockerfile ile Calistirma

```bash
# Imaj olusturma
docker build -t telco-churn-api .

# Konteyner calistirma
docker run -p 8000:8000 telco-churn-api
```

### Konteyneri Durdurma

```bash
docker-compose down
```

---

## Testler

### API Testi (Manuel)

API calistiktan sonra asagidaki komutu calistirin:

```bash
curl http://localhost:8000/
```

Beklenen cevap:

```json
{"message": "Telco Churn Prediction API is running"}
```

### Model Dogrulugu Testi

Egitim scripti calistirildiginda dogruluk (accuracy) degeri terminale yazdirilir:

```bash
python src/train.py
# Cikti: Accuracy: 0.80xx
```

### Docker Ortaminda Test

```bash
docker-compose up --build -d
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customerID":"test","gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":70.35,"TotalCharges":"840.2"}'
docker-compose down
```

---

## Katki

1. Repoyu fork'layin.
2. Yeni bir branch olusturun: `git checkout -b feature/yeni-model`
3. Degisikliklerinizi commit edin: `git commit -m "Yeni model eklendi"`
4. Branch'inizi push edin: `git push origin feature/yeni-model`
5. Pull Request aciniz.

---

## Lisans

Bu proje MIT Lisansi ile lisanslanmistir. Detaylar icin `LICENSE` dosyasina bakiniz.
