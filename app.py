import streamlit as st
import pandas as pd
import joblib

st.title("Regressi Penunjung Borobudur")
st.markdown("Aplikasi machine learning regression untuk menghitung pengunjung di Borobudur")

model_forest=joblib.load("model_forest.joblib")

hari_type = st.pills("Hari Type",["weekday","weekend"],default="weekend")
musim = st.pills("Musim",["kemarau","hujan"],default="hujan")
suhu_rata_rata = st.slider("Suhu Rata Rata",10.0,60.0,25.0)
ada_event_budaya = st.pills("Ada Event Budaya",["ya","tidak"],default="ya")
harga_tiket_ribu = st.slider("Harga Tiket Ribu",50.0,100.0,70.0)

if st.button("prediksi"):
    data_baru = pd.DataFrame(
        [[hari_type, musim, suhu_rata_rata, harga_tiket_ribu, ada_event_budaya]],
        columns=["hari_type", "musim", "suhu_rata_rata", "harga_tiket_ribu", "ada_event_budaya"]
    )

    prediksi = model_forest.predict(data_baru)[0]
    st.success(f"Model memprediksi banyak pengunjung {prediksi:.0f}")
    st.balloons()



