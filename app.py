import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================================================
# Load Model dan Encoder
# ======================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("powerconsumption_knn_model.pkl")
        encoder = joblib.load("label_encoder.pkl")
        return model, encoder
    except Exception as e:
        st.error(
            "Gagal memuat model. Penyebab umum: ketidakcocokan serialisasi (mis. objek yang dibuat dengan numba)\n"
            "Solusi cepat: pasang paket `numba` yang sama dengan lingkungan saat model dibuat, atau re-export model tanpa objek ter-jit.\n"
            f"Detail error: {e}"
        )
        raise

model, encoder = load_artifacts()

# ======================================================
# UI
# ======================================================
st.set_page_config(page_title="Power Consumption Classifier", layout="wide")

st.title("🔌 Power Consumption Season Classification")
st.markdown("""
Aplikasi ini mengklasifikasikan **pola konsumsi listrik rumah tangga**
ke dalam **Musim Hangat (Warm Season)** atau **Musim Dingin (Cold Season)**
berdasarkan **data time series konsumsi daya listrik**.
""")

# ======================================================
# Sidebar Input
# ======================================================
st.sidebar.header("📥 Input Data Time Series")
st.sidebar.write("Masukkan **144 nilai** konsumsi daya listrik")

# Opsi: paste 144 nilai sekaligus (dipisah spasi/newline) atau isi manual per timestep
paste_text = st.sidebar.text_area(
    "Tempel 144 nilai (pisahkan dengan spasi atau newline)",
    value="",
    height=140,
)

parsed_series = None
if paste_text and paste_text.strip():
    tokens = paste_text.strip().split()
    try:
        vals = [float(t) for t in tokens]
        if len(vals) != 144:
            st.sidebar.error(f"Jumlah nilai yang ditempel: {len(vals)} — dibutuhkan 144 nilai.")
        else:
            parsed_series = np.array(vals)
    except ValueError:
        st.sidebar.error("Gagal mengurai beberapa nilai. Pastikan angka dipisah spasi atau newline.")

if parsed_series is None:
    input_series = []
    for i in range(144):
        value = st.sidebar.number_input(
            f"Timestep {i+1}",
            value=0.0,
            step=0.01,
            format="%.2f",
            key=f"ts_{i}"
        )
        input_series.append(value)

    input_series = np.array(input_series)
else:
    input_series = parsed_series

# ======================================================
# Visualisasi Input
# ======================================================
st.subheader("📈 Visualisasi Time Series")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(input_series)
ax.set_xlabel("Time Index")
ax.set_ylabel("Power Consumption")
ax.grid(True)
st.pyplot(fig)

# ======================================================
# Prediksi
# ======================================================
if st.button("🔍 Prediksi Musim"):
    X_input = input_series.reshape(1, -1)  # FLATTEN untuk KNN
    prediction = model.predict(X_input)[0]
    predicted_label = encoder.inverse_transform([prediction])[0]
    readable_label = str(predicted_label)

    st.subheader("✅ Hasil Prediksi")

    # Tampilkan hasil prediksi dengan deteksi kata kunci 'warm' / 'hangat'
    if "warm" in readable_label.lower() or "hangat" in readable_label.lower():
        st.success("🌞 Musim Hangat (Warm Season)")
    else:
        st.info("❄️ Musim Dingin (Cold Season)")

    st.write(f"**Prediksi (original):** {predicted_label}")

    # Probabilitas (opsional) — map kolom proba ke nama kelas sebenarnya
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]
        classes = getattr(model, "classes_", None)
        if classes is None:
            # fallback ke asumsi indeks 0=Cold, 1=Warm
            proba_dict = {"Cold Season": float(proba[0]), "Warm Season": float(proba[1])}
        else:
            try:
                label_names = [str(encoder.inverse_transform([c])[0]) for c in classes]
            except Exception:
                label_names = [str(c) for c in classes]
            proba_dict = {name: float(proba[i]) for i, name in enumerate(label_names)}

        st.write("### 📊 Probabilitas Kelas")
        st.bar_chart(proba_dict)

# ======================================================
# Footer
# ======================================================
st.markdown("---")
st.caption("Proyek Data Sains – CRISP-DM | Dataset PowerCons")
