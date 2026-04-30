import gradio as gr
import requests

# API'nin çalıştığı adres — gerekirse değiştirilebilir.
API_URL = "http://localhost:8000/predict"


def tahmin_et(
    gender, SeniorCitizen, Partner, Dependents, tenure,
    PhoneService, MultipleLines, InternetService, OnlineSecurity,
    OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
    StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges
):
    """Formdaki müşteri bilgilerini API'ye gönderir ve sonucu döndürür."""

    veri = {
        "customerID": "ui-test",
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": str(TotalCharges),
    }

    try:
        yanit = requests.post(API_URL, json=veri, timeout=5)
        yanit.raise_for_status()
        sonuc = yanit.json()

        tahmin = sonuc["prediction"]
        olasilik = sonuc["churn_probability"]

        etiket = "⚠️ Müşteri kaybedilme riski YÜKSEK" if tahmin == 1 else "✅ Müşteri kaybedilme riski DÜŞÜK"
        return f"{etiket}\n\nChurn Olasılığı: {olasilik:.2%}"

    except requests.exceptions.ConnectionError:
        return "❌ API'ye bağlanılamadı. Lütfen önce API'yi başlatın:\nuvicorn api.main:app --host 0.0.0.0 --port 8000"
    except Exception as e:
        return f"❌ Hata: {str(e)}"


# Arayüz bileşenleri
with gr.Blocks(title="Telco Churn Tahmini", theme=gr.themes.Soft()) as arayuz:

    gr.Markdown("# 📊 Telco Müşteri Kaybı Tahmini")
    gr.Markdown("Müşteri bilgilerini girin ve modelin churn tahminini görün.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Müşteri Bilgileri")
            gender = gr.Dropdown(["Male", "Female"], label="Cinsiyet", value="Male")
            SeniorCitizen = gr.Checkbox(label="Yaşlı Vatandaş (65+)", value=False)
            Partner = gr.Dropdown(["Yes", "No"], label="Eş / Partner", value="No")
            Dependents = gr.Dropdown(["Yes", "No"], label="Bakmakla Yükümlü Kişi", value="No")
            tenure = gr.Slider(0, 72, value=12, step=1, label="Hizmet Süresi (ay)")

        with gr.Column():
            gr.Markdown("### Hizmet Bilgileri")
            PhoneService = gr.Dropdown(["Yes", "No"], label="Telefon Hizmeti", value="Yes")
            MultipleLines = gr.Dropdown(
                ["Yes", "No", "No phone service"], label="Çoklu Hat", value="No"
            )
            InternetService = gr.Dropdown(
                ["DSL", "Fiber optic", "No"], label="İnternet Hizmeti", value="DSL"
            )
            OnlineSecurity = gr.Dropdown(
                ["Yes", "No", "No internet service"], label="Çevrimiçi Güvenlik", value="No"
            )
            OnlineBackup = gr.Dropdown(
                ["Yes", "No", "No internet service"], label="Çevrimiçi Yedekleme", value="No"
            )

        with gr.Column():
            gr.Markdown("### Ek Hizmetler ve Ödeme")
            DeviceProtection = gr.Dropdown(
                ["Yes", "No", "No internet service"], label="Cihaz Koruma", value="No"
            )
            TechSupport = gr.Dropdown(
                ["Yes", "No", "No internet service"], label="Teknik Destek", value="No"
            )
            StreamingTV = gr.Dropdown(
                ["Yes", "No", "No internet service"], label="TV Yayını", value="No"
            )
            StreamingMovies = gr.Dropdown(
                ["Yes", "No", "No internet service"], label="Film Yayını", value="No"
            )
            Contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"], label="Sözleşme Tipi", value="Month-to-month"
            )
            PaperlessBilling = gr.Dropdown(["Yes", "No"], label="Kağıtsız Fatura", value="Yes")
            PaymentMethod = gr.Dropdown(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                label="Ödeme Yöntemi",
                value="Electronic check"
            )
            MonthlyCharges = gr.Number(label="Aylık Ücret ($)", value=29.85)
            TotalCharges = gr.Number(label="Toplam Ücret ($)", value=29.85)

    tahmin_butonu = gr.Button("🔍 Tahmin Et", variant="primary")
    sonuc = gr.Textbox(label="Sonuç", lines=3)

    tahmin_butonu.click(
        fn=tahmin_et,
        inputs=[
            gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, MultipleLines, InternetService, OnlineSecurity,
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges
        ],
        outputs=sonuc
    )

    gr.Markdown("---\n*P2P Veri Bilimi Challenge — Telco Churn Projesi*")


if __name__ == "__main__":
    arayuz.launch()
