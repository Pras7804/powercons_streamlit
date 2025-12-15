import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================================================
# WAJIB: Definisi class SimpleKNN (HARUS ADA)
# ======================================================
class SimpleKNN:
    def __init__(self, X_train, y_train, k=5):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.linalg.norm(self.X_train - x, axis=1)
            idx = np.argsort(dists)[:self.k]
            votes = self.y_train[idx]
            preds.append(np.bincount(votes).argmax())
        return np.array(preds)

# ======================================================
# Load Model dan Encoder
# ======================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("powerconsumption_knn_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_artifacts()

# ======================================================
# UI CONFIG
# ======================================================
st.set_page_config(
    page_title="Power Consumption Classifier",
    layout="wide"
)

st.title("🔌 Power Consumption Season Classification")

st.markdown("""
Aplikasi ini mengklasifikasikan **pola konsumsi listrik rumah tangga**
ke dalam **Musim Hangat (Warm Season)** atau **Musim Dingin (Cold Season)**  
berdasarkan **data time series konsumsi daya listrik (144 timestep)**.
""")

# ======================================================
# Sidebar Input
# ======================================================
st.sidebar.header("📥 Input Data Time Series")
st.sidebar.write("Masukkan **144 nilai konsumsi daya listrik**")

paste_text = st.sidebar.text_area(
    "Tempel 144 nilai (pisahkan spasi atau newline)",
    height=150
)

parsed_series = None
if paste_text.strip():
    try:
        values = [float(v) for v in paste_text.split()]
        if len(values) != 144:
            st.sidebar.error(f"Jumlah nilai: {len(values)} (harus 144)")
        else:
            parsed_series = np.array(values)
    except ValueError:
        st.sidebar.error("Pastikan semua input berupa angka")

if parsed_series is None:
    input_series = []
    for i in range(144):
        val = st.sidebar.number_input(
            f"Timestep {i+1}",
            value=0.0,
            step=0.01,
            format="%.2f",
            key=f"ts_{i}"
        )
        input_series.append(val)
    input_series = np.array(input_series)
else:
    input_series = parsed_series

# ======================================================
# Visualisasi
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
    X_input = input_series.reshape(1, -1)

    pred_idx = model.predict(X_input)[0]
    pred_label = encoder.inverse_transform([pred_idx])[0]

    st.subheader("✅ Hasil Prediksi")

    if "warm" in str(pred_label).lower() or "hangat" in str(pred_label).lower():
        st.success("🌞 Musim Hangat (Warm Season)")
    else:
        st.info("❄️ Musim Dingin (Cold Season)")

    st.write(f"**Label asli:** {pred_label}")

# ======================================================
# Footer
# ======================================================
st.markdown("---")
st.caption("Proyek Data Sains – CRISP-DM | Dataset PowerCons")
