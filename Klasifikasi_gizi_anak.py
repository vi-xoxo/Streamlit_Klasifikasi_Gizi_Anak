import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
pipeline = joblib.load("model/pipeline_status_gizi.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

st.set_page_config(page_title="Prediksi Status Gizi Anak", layout="centered")

st.title("Prediksi Status Gizi Anak")
st.write("Masukkan data anak, lalu klik Prediksi.")

# =========================
# INPUT
# =========================
col1, col2 = st.columns(2)

with col1:
    age_months = st.number_input("Usia (bulan)", min_value=0.0, step=1.0)
    weight_kg = st.number_input("Berat (kg)", min_value=0.0, step=0.1)
    muac_cm = st.number_input("Lingkar Lengan Atas MUAC (cm)", min_value=0.0, step=0.1)

with col2:
    height_cm = st.number_input("Tinggi (cm)", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi"):
    input_df = pd.DataFrame([{
        "age_months": age_months,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "muac_cm": muac_cm,
        "bmi": bmi
    }])

    pred_encoded = pipeline.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    st.subheader("Hasil Prediksi")
    st.success(f"Status gizi: **{pred_label.upper()}**")

    # Probabilitas
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(input_df)[0]
        proba_df = pd.DataFrame({
            "Status Gizi": label_encoder.classes_,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False)

        st.write("Probabilitas per kelas")
        st.dataframe(proba_df, use_container_width=True)

        st.bar_chart(proba_df.set_index("Status Gizi"))

# =========================
# INFO
# =========================
st.caption("Deployment model Random Forest untuk klasifikasi status gizi anak.")
