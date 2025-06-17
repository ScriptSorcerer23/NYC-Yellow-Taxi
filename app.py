import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image

# 🔧 MUST BE FIRST
st.set_page_config(page_title="NYC Taxi Predictor", layout="wide")

# =======================
# 📌 Custom CSS Styling
# =======================

st.markdown("""
<style>
/* 🌃 Base Gradient Background */
html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    background: linear-gradient(-45deg, yellow, #2c2c2c, #1a1a1a, black);
    background-size: 400% 400%;
    animation: cityLightsFlow 100s ease-in-out infinite;
    color: #f0f0f0 !important;
}

/* 🔲 Sidebar background */
[data-testid="stSidebar"] {
    background-color: #2c2c2c !important;
    border-right: 2px solid #444;
}

/* 📌 Sidebar Text Styling */
[data-testid="stSidebar"] .element-container div,
[data-testid="stSidebar"] .element-container p,
[data-testid="stSidebar"] .element-container span {
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    text-align: center;
}

/* 📍 Sidebar header elements (more specific selector) */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    text-align: center;
}

/* 📌 Main Markdown text styling */
.markdown-text-container p,
.markdown-text-container span,
.markdown-text-container strong {
    color: white !important;
    font-weight: 700 !important;
}

/* ✏️ Input Fields Improvements (merged and refined) */
.stNumberInput label, .stTextInput label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
}

.stNumberInput input, .stTextInput input {
    background-color: #444 !important;
    color: #ffffff !important;
    border: 2px solid #777 !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
    font-size: 1.1rem !important;
}

/* Optional: Give input boxes some box-shadow to pop */
.stNumberInput input:focus, .stTextInput input:focus {
    outline: none !important;
    border: 2px solid #FFD700 !important;
    box-shadow: 0 0 8px rgba(255, 215, 0, 0.6);
}

/* 🔘 Radio buttons – Fare Amount / Payment Type */
.stRadio label,
div[data-baseweb="radio"] div[role="radio"] {
    font-weight: 800 !important;
    color: #ffffff !important;
    font-size: 1.4rem !important;
}

/* 🏷️ Headings & Labels */
h1, h2, h3, label {
    color: #f0f0f0 !important;
    font-weight: bold;
}
h1 { font-size: 3rem !important; }
h2 { font-size: 2.2rem !important; }

/* 🖼️ Images */
img {
    border-radius: 12px;
    box-shadow: 0 0 10px rgba(255,255,255,0.1);
}

/* 🚀 Predict Button (target by key for clarity) */
div[data-testid="stKey-predict"] button { /* Targeting by key directly */
    background-color: #FFD700;
    color: black;
    font-weight: bold;
    border-radius: 12px;
    font-size: 2rem;
    height: 3rem;
    width: 100%;
    transition: 0.3s ease;
}
div[data-testid="stKey-predict"] button:hover {
    background-color: #FFC800;
    transform: scale(1.03);
}

/* 📑 Summary Button */
div[data-testid="stKey-summary"] button {
    background-color: #1E90FF;
    color: white;
    font-weight: bold;
    border-radius: 12px;
    font-size: 1.6rem;
    height: 3rem;
    width: 100%;
    transition: 0.3s ease;
}
div[data-testid="stKey-summary"] button:hover {
    background-color: #187BCD;
    transform: scale(1.03);
}

/* 📊 Visualize Button */
div[data-testid="stKey-visual"] button {
    background-color: #32CD32;
    color: white;
    font-weight: bold;
    border-radius: 12px;
    font-size: 1.6rem;
    height: 3rem;
    width: 100%;
    transition: 0.3s ease;
}
div[data-testid="stKey-visual"] button:hover {
    background-color: #28A428;
    transform: scale(1.03);
}

/* 🔁 Background Flow Animation */
@keyframes cityLightsFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
/* 🔘 Radio buttons – Fare Amount / Payment Type */
/* Targeting the actual label text more directly within stRadio component */
div[data-testid="stRadio"] label p { /* Target the <p> tag inside the label which holds the text */
    font-weight: 800 !important;
    color: #ffffff !important; /* Ensure this is pure white and takes precedence */
    font-size: 1.4rem !important;
    -webkit-text-fill-color: #ffffff !important; /* For some browser compatibility with text color overrides */
    opacity: 1 !important; /* Ensure full opacity */
}

/* Also ensure the radio button circles themselves are visible if they are affected */
div[data-baseweb="radio"] div[role="radio"] {
    border-color: #ffffff !important; /* Ensure the outer circle border is white */
    background-color: transparent !important; /* Keep background transparent or set as needed */
}

div[data-baseweb="radio"] div[role="radio"][aria-checked="true"] {
    background-color: #FFD700 !important; /* Gold color for checked state */
    border-color: #FFD700 !important; /* Gold color for checked state border */
}
</style>
""", unsafe_allow_html=True)


# =======================
# 📦 Load Models
# =======================
try:
    with open("logistic_regression_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open("decision_tree_model.pkl", "rb") as f:
        dt_model = pickle.load(f)
    with open("kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    with open("linear_regression_fare.pkl", "rb") as f:
        linreg_model = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error loading model: {e}. Make sure all .pkl files are in the same directory.")
    st.stop() # Stop execution if models cannot be loaded

# =======================
# 📊 Sidebar Visual Insights
# =======================
st.sidebar.markdown('<div style="text-align:center; font-weight:bold; margin-bottom:10px;">Model Comparison Overview</div>', unsafe_allow_html=True)
try:
    st.sidebar.image("model_comparison.png", use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("`model_comparison.png` not found. Please ensure image file exists.")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown('<div style="text-align:center; font-weight:bold; margin-bottom:10px;">What influences tips the most?</div>', unsafe_allow_html=True)
try:
    st.sidebar.image("features_tips.png", use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("`features_tips.png` not found. Please ensure image file exists.")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown('<div style="text-align:center; font-weight:bold; margin-bottom:10px;">Most people travel alone!</div>', unsafe_allow_html=True)
try:
    st.sidebar.image("passenger.png", use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("`passenger.png` not found. Please ensure image file exists.")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown('<div style="text-align:center; font-weight:bold; margin-bottom:10px;">Yellow cab demands in NYC rise between 1 PM and 7 PM, peaking around 6 PM</div>', unsafe_allow_html=True)
try:
    st.sidebar.image("hours_demand.png", use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("`hours_demand.png` not found. Please ensure image file exists.")


# =======================
# 🧠 Main App Content
# =======================
st.markdown("""
    <h1 style='text-align: center; font-size: 2.8rem;'>🚖 TaxiTeller NYC</h1>
    <p style='text-align: left; font-size: 1.2rem; margin-top: 1.5rem;'>
        Enter ride details and predict the <strong>payment type</strong> and <strong>cluster group</strong>.<br>
        Also explore model behavior and clusters below.
    </p>
""", unsafe_allow_html=True)

predict_choice = st.radio("What would you like to predict?", ["Fare Amount", "Payment Type"], index=1, help="Select 'Fare Amount' to predict the cost of a trip, or 'Payment Type' to predict how the trip will be paid for.")

# Initialize all input variables with sensible defaults
passenger_count = 1
trip_distance = 2.5
fare_amount = 10.0
tip_amount = 2.5
tolls_amount = 0.3
vendor_id = 0.5 # Consider making this an input if meaningful
rate_code = 0.5 # Consider making this an input if meaningful
trip_time = 2.0 # Consider making this an input if meaningful (e.g., in minutes)
airport_fee = 0.0 # Consider making this an input if meaningful
pickup_lat = 13.3 # Consider making this an input if meaningful
extras = 1.0 # Consider making this an input if meaningful


if predict_choice == "Fare Amount":
    st.markdown("#### Enter ride details for fare prediction")
    # Make passenger_count an input for fare prediction if used by model
    passenger_count = st.number_input("👥 Passenger Count", min_value=1, max_value=6, value=1, key="fare_passenger_count")
    trip_distance = st.number_input("📏 Trip Distance (miles)", min_value=0.0, value=2.5, step=0.1, key="fare_trip_distance")
    tip_amount = st.number_input("💸 Tip Amount ($)", min_value=0.0, value=2.5, step=0.5, key="fare_tip_amount")
    tolls_amount = st.number_input("🛣️ Tolls Amount ($)", min_value=0.0, value=0.3, step=0.1, key="fare_tolls_amount")

    # Assuming these fixed values are indeed part of the fare prediction model's input
    input_data = np.array([[passenger_count, trip_distance, tip_amount, tolls_amount, extras]])

else: # predict_choice == "Payment Type"
    st.markdown("#### Enter ride details for payment type and cluster prediction")
    passenger_count = st.number_input("👥 Passenger Count", min_value=1, max_value=6, value=1, key="payment_passenger_count")
    trip_distance = st.number_input("📏 Trip Distance (miles)", min_value=0.0, value=2.5, step=0.1, key="payment_trip_distance")
    fare_amount = st.number_input("💵 Fare Amount ($)", min_value=0.0, value=10.0, step=0.5, key="payment_fare_amount")

    # Assuming these fixed values are indeed part of the payment type prediction model's input
    input_data = np.array([[passenger_count, trip_distance, fare_amount, vendor_id, rate_code,
                             trip_time, airport_fee, tip_amount, pickup_lat, extras]])

payment_mapping = {
    1: "Credit card", 2: "Cash", 3: "No charge",
    4: "Dispute", 5: "Unknown", 6: "Voided trip"
}
cluster_descriptions = {
    0: "Most common trips – short distance, moderate fare, mostly card payments",
    1: "Sparse cluster",
    2: "Sparse cluster",
    3: "Outlier cluster – extremely long trip durations (~1400 minutes!)",
    4: "High-value trips – long distance, higher fares, more tolls and tips"
}

# Initialize session state for buttons if not already present
if "trigger_predict" not in st.session_state:
    st.session_state.trigger_predict = False
if "trigger_summary" not in st.session_state:
    st.session_state.trigger_summary = False
if "trigger_visual" not in st.session_state:
    st.session_state.trigger_visual = False

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    predict_button = st.button("🚀 Predict", key="predict")
    if predict_button:
        st.session_state.trigger_predict = True
    st.markdown("""
        <style>
            div[data-testid="stButton"] > button:first-child {
                background-color: #32CD32;
                color: black;
                size
            }
            div[data-testid="stButton"] > button:first-child:hover {
                background-color: #1E90FF;
            }
        </style>
    """, unsafe_allow_html=True)

with col2:
    summary_button = st.button("📑 See Cluster Summary", key="summary")
    if summary_button:
        st.session_state.trigger_summary = True
    st.markdown("""
        <style>
            div[data-testid="stButton"] > button:nth-child(2) {
                background-color: #1E90FF;
                color: white;
            }
            div[data-testid="stButton"] > button:nth-child(2):hover {
                background-color: #187BCD;
            }
        </style>
    """, unsafe_allow_html=True)

with col3:
    visual_button = st.button("📊 Visualize Clusters", key="visual")
    if visual_button:
        st.session_state.trigger_visual = True
    st.markdown("""
        <style>
            div[data-testid="stButton"] > button:nth-child(3) {
                background-color: #32CD32;
                color: white;
            }
            div[data-testid="stButton"] > button:nth-child(3):hover {
                background-color: #28A428;
            }
        </style>
    """, unsafe_allow_html=True)
# =======================
# ✅ Button Logic
# =======================
if st.session_state.trigger_predict:
    st.session_state.trigger_predict = False
    st.subheader("🔍 Prediction Results")

    if predict_choice == "Fare Amount":
        if input_data is not None:
            linreg_pred = linreg_model.predict(input_data)[0]
            st.markdown(f"**🧾 Linear Regression (Fare Prediction):** `${linreg_pred:.2f}`")
        else:
            st.error("Missing input for fare prediction.")
    else: # Payment Type
        if input_data is not None:
            lr_pred = lr_model.predict(input_data)[0]
            dt_pred = dt_model.predict(input_data)[0]
            cluster_pred = kmeans_model.predict(input_data)[0]

            lr_proba = lr_model.predict_proba(input_data)[0]
            dt_proba = dt_model.predict_proba(input_data)[0]

            st.markdown(f"**🤖 Logistic Regression Prediction:** `{payment_mapping.get(lr_pred, 'Unknown')}` (Probability: **{max(lr_proba):.2f}**)")
            # Using st.metric or a custom visual if a bar is desired, not st.progress
            # st.progress(float(max(lr_proba))) # Removed as it implies ongoing process

            st.markdown(f"**🌳 Decision Tree Prediction:** `{payment_mapping.get(dt_pred, 'Unknown')}` (Probability: **{max(dt_proba):.2f}**)")
            # st.progress(float(max(dt_proba))) # Removed

            st.markdown(f"**🔢 KMeans Cluster:** `{cluster_pred}` – {cluster_descriptions.get(cluster_pred, 'Unknown cluster')}")
        else:
            st.error("Missing input for payment prediction.")

# 📑 Cluster Summary
if st.session_state.trigger_summary:
    st.session_state.trigger_summary = False
    st.subheader("📑 Cluster Summary")
    for cluster_id, description in cluster_descriptions.items():
        st.markdown(f"**Cluster {cluster_id}:** {description}")

# 📊 Visualize Clusters (Placeholder)
if st.session_state.trigger_visual:
    st.session_state.trigger_visual = False
    st.subheader("Cluster Visualization")
    try:
        cluster_img = Image.open("cluster_plot.png")
        st.image(cluster_img, caption="KMeans Clustering on NYC Taxi Data", use_container_width=True)
    except FileNotFoundError:
        st.error("`cluster_plot.png` not found. Please ensure image file exists for visualization.")

st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)