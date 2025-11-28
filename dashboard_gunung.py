# dashboard_gunung.py
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# ---------------------------
# Load dataset dan model
# ---------------------------
df = pd.read_csv("dataset_gunung_final.csv")

# Load model MLP tanpa compile (menghindari error mse)
model = load_model("mlp_gunung_model.h5", compile=False)

# ---------------------------
# Sidebar: Input User
# ---------------------------
st.sidebar.header("Input Preferensi Pendakian")

lokasi_user = st.sidebar.selectbox("Pilih Lokasi Anda:", options=df["Province"].unique())
pengalaman = st.sidebar.selectbox("Tingkat Pengalaman:", ["Beginner", "Intermediate", "Advanced"])
durasi_max = st.sidebar.slider("Durasi Pendakian Maksimal (jam):", min_value=1, max_value=12, value=6)
tingkat_kesulitan = st.sidebar.multiselect(
    "Tingkat Kesulitan:", options=df["difficulty_level"].unique(), default=df["difficulty_level"].unique()
)
jarak_max = st.sidebar.slider("Jarak Maksimal dari Lokasi (km):", min_value=1, max_value=50, value=10)

submit = st.sidebar.button("Cari Rekomendasi")

# ---------------------------
# Fungsi Filter & Ranking
# ---------------------------
def filter_gunung(df, lokasi, pengalaman, durasi, kesulitan, jarak):
    filtered = df[
        (df["Province"] == lokasi) &
        (df["recommended_for"] == pengalaman) &
        (df["hiking_duration_hours"] <= durasi) &
        (df["difficulty_level"].isin(kesulitan)) &
        (df["distance_km"] <= jarak)
    ]
    return filtered

def predict_score(filtered_df):
    if filtered_df.empty:
        return filtered_df
    X = filtered_df[["elevation_m", "hiking_duration_hours", "distance_km"]].values  # numpy array
    scores = model.predict(X).flatten()
    filtered_df = filtered_df.copy()
    filtered_df["mlp_score"] = scores
    filtered_df = filtered_df.sort_values(by="mlp_score", ascending=False)
    return filtered_df

# ---------------------------
# Main
# ---------------------------
st.title("Sistem Rekomendasi Gunung Pendakian Pulau Jawa")

if submit:
    hasil_filter = filter_gunung(df, lokasi_user, pengalaman, durasi_max, tingkat_kesulitan, jarak_max)
    hasil_rank = predict_score(hasil_filter)

    if hasil_rank.empty:
        st.warning("Maaf, tidak ada gunung yang sesuai dengan preferensi Anda.")
    else:
        st.success(f"Ditemukan {len(hasil_rank)} gunung sesuai preferensi Anda!")
        for _, row in hasil_rank.iterrows():
            st.markdown(f"### {row['Name']} ({row['Province']})")
            st.markdown(f"**Kesulitan:** {row['difficulty_level']} | **Durasi:** {row['hiking_duration_hours']} jam | **Jarak:** {row['distance_km']} km")
            st.markdown(f"[Lihat di Maps]({row['source_url']})")
            if pd.notna(row.get('image_url', None)):
                st.image(row['image_url'], width=400)
