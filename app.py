import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef4ff 0%, #eefcf8 45%, #f8efff 100%);
}

.block-container {
    max-width: 980px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #1f2a44 !important;
    font-weight: 800 !important;
    margin-bottom: 0.25rem !important;
}

h3 {
    color: #243b6b !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown, .stCaption {
    color: #2c3550 !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > div {
    background: rgba(255, 255, 255, 0.92) !important;
    border: 1px solid #d6dcff !important;
    border-radius: 14px !important;
}

.stTextInput input,
.stNumberInput input {
    background: rgba(255, 255, 255, 0.92) !important;
}

.stButton > button {
    width: 100%;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.8rem 1rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: white !important;
    background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%) !important;
    box-shadow: 0 10px 24px rgba(79, 70, 229, 0.25);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4338ca 0%, #0891b2 100%) !important;
}

.section-card {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(155, 170, 255, 0.28);
    border-radius: 22px;
    padding: 20px 20px 8px 20px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(81, 102, 255, 0.10);
    backdrop-filter: blur(8px);
}

.hero-box {
    background: linear-gradient(135deg, rgba(79,70,229,0.10), rgba(6,182,212,0.10), rgba(168,85,247,0.10));
    border: 1px solid rgba(120, 131, 255, 0.22);
    border-radius: 24px;
    padding: 20px 22px;
    margin-bottom: 18px;
}

.badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    color: #4338ca;
    background: rgba(79, 70, 229, 0.10);
    margin-bottom: 0.75rem;
}

.metric-box {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(155, 170, 255, 0.22);
    border-radius: 18px;
    padding: 6px;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(155, 170, 255, 0.16);
    border-radius: 18px;
    padding: 14px;
}

[data-testid="stAlert"] {
    border-radius: 16px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <div class="badge">ML Classification Demo</div>
    <h1>Customer Churn Prediction</h1>
    <p>Enter customer details to estimate churn risk using a trained Gradient Boosting pipeline.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    df = pd.read_excel("Telco_customer_churn.xlsx")

    target = "Churn Value"

    drop_cols = [
        "CustomerID",
        "Count",
        "Churn Label",
        "Churn Score",
        "CLTV",
        "Churn Reason"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols + [target]).copy()
    y = df[target].copy()

    X = X.replace([" ", "", "NA", "N/A", "null", "None"], np.nan)
    X = X.infer_objects(copy=False)

    force_numeric = [
        "Zip Code",
        "Latitude",
        "Longitude",
        "Tenure Months",
        "Monthly Charges",
        "Total Charges"
    ]
    force_numeric = [c for c in force_numeric if c in X.columns]

    for col in force_numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    for col in cat_cols:
        X[col] = X[col].astype(str)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(random_state=42))
    ])

    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_order = train_model()

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Location Details")
col1, col2 = st.columns(2)

with col1:
    country = st.selectbox("Country", ["United States"])
    state = st.text_input("State", "California")
    city = st.text_input("City", "San Diego")

with col2:
    zip_code = st.number_input("Zip Code", min_value=1, value=92101)
    latitude = st.number_input("Latitude", value=32.7157, format="%.6f")
    longitude = st.number_input("Longitude", value=-117.1611, format="%.6f")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Customer Profile")
col3, col4 = st.columns(2)

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])

with col4:
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Services and Billing")
col5, col6 = st.columns(2)

with col5:
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

with col6:
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer automatic",
            "Credit card automatic"
        ]
    )
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

st.markdown("</div>", unsafe_allow_html=True)

input_data = {
    "Country": country,
    "State": state,
    "City": city,
    "Zip Code": zip_code,
    "Lat Long": f"{latitude}, {longitude}",
    "Latitude": latitude,
    "Longitude": longitude,
    "Gender": gender,
    "Senior Citizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure,
    "Phone Service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges
}

if st.button("Predict Churn"):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_order)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    m1, m2 = st.columns(2)

    with m1:
        st.metric("Predicted Churn", "Yes" if prediction == 1 else "No")

    with m2:
        st.metric("Churn Probability", f"{probability:.2%}")

    if probability >= 0.50:
        st.error("High churn risk detected. This customer is more likely to leave.")
    else:
        st.success("Low churn risk detected. This customer is less likely to leave.")
