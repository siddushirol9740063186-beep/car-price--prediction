import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Car Price Prediction", layout="centered")

# ---------------- DARK UI ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3 {
    color: white;
}
.stButton>button {
    background-color: #1f2937;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- HEADER ----------------
st.title("🚗 Car Price Prediction")
st.write("Predict resale price using ML")

st.subheader("Enter Car Details")

# ---------------- INPUTS ----------------
year = st.slider("Year of Purchase", 2000, 2025, 2015)
present_price = st.number_input("Present Price (Lakhs)", 0.0, 50.0, 5.0)
kms_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Number of Owners", [0, 1, 2, 3])

# ---------------- ENCODING ----------------
fuel_map = {"Petrol": 2, "Diesel": 1, "CNG": 0}
seller_map = {"Dealer": 0, "Individual": 1}
transmission_map = {"Manual": 1, "Automatic": 0}

car_age = 2025 - year

input_data = np.array([[present_price, kms_driven, fuel_map[fuel],
                        seller_map[seller], transmission_map[transmission],
                        owner, car_age]])

# ---------------- PREDICT ----------------
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹ {round(prediction[0], 2)} Lakhs")

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("Built with ❤️ using Machine Learning & Streamlit")
