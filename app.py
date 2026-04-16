import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4f7ff 0%, #eefafc 45%, #f9f0ff 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 950px;
}

h1 {
    color: #1f2a44 !important;
    font-weight: 800 !important;
}

h3 {
    color: #243b6b !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > div {
    background-color: rgba(255,255,255,0.92) !important;
    border: 1px solid #d6dcff !important;
    border-radius: 12px !important;
}

.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background-color: rgba(255,255,255,0.92) !important;
}

.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.2rem !important;
    font-weight: 700 !important;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4338ca, #0891b2);
}

[data-testid="stAlert"] {
    border-radius: 14px !important;
}

.section-card {
    background: rgba(255,255,255,0.78);
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(88, 108, 255, 0.10);
    border: 1px solid rgba(155, 170, 255, 0.25);
}
</style>
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

input_data = {
    "Country": st.selectbox("Country", ["United States"]),
    "State": st.text_input("State", "California"),
    "City": st.text_input("City", "San Diego"),
    "Zip Code": st.number_input("Zip Code", min_value=1, value=92101),
    "Lat Long": st.text_input("Lat Long", "32.7157, -117.1611"),
    "Latitude": st.number_input("Latitude", value=32.7157, format="%.6f"),
    "Longitude": st.number_input("Longitude", value=-117.1611, format="%.6f"),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "Senior Citizen": st.selectbox("Senior Citizen", ["Yes", "No"]),
    "Partner": st.selectbox("Partner", ["Yes", "No"]),
    "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
    "Tenure Months": st.number_input("Tenure Months", min_value=0, max_value=100, value=12),
    "Phone Service": st.selectbox("Phone Service", ["Yes", "No"]),
    "Multiple Lines": st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
    "Internet Service": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "Online Security": st.selectbox("Online Security", ["Yes", "No", "No internet service"]),
    "Online Backup": st.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
    "Device Protection": st.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
    "Tech Support": st.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
    "Streaming TV": st.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
    "Streaming Movies": st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
    "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "Paperless Billing": st.selectbox("Paperless Billing", ["Yes", "No"]),
    "Payment Method": st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer automatic", "Credit card automatic"]
    ),
    "Monthly Charges": st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0),
    "Total Charges": st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)
}

if st.button("Predict Churn"):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_order]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write("Predicted Churn:", "Yes" if prediction == 1 else "No")
    st.write("Churn Probability:", f"{probability:.2%}")

    if probability >= 0.5:
        st.error("High churn risk")
    else:
        st.success("Low churn risk")        
