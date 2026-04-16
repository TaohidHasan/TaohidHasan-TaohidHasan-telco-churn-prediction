import os
from pathlib import Path

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
    page_icon="📉",
    layout="wide"
)

POSSIBLE_DATA_FILES = [
    "Telco_customer_churn.xlsx",
    "Telcocustomerchurn.xlsx",
    "telco_customer_churn.xlsx",
    "Telco Customer Churn.xlsx"
]

st.markdown("""
<style>
:root {
    --vf-red: #e60000;
    --vf-red-dark: #b80000;
    --vf-red-soft: #fff1f1;
    --vf-bg: #f7f8fa;
    --vf-card: #ffffff;
    --vf-card-soft: #fcfcfd;
    --vf-border: #e6e8ee;
    --vf-text: #1f2937;
    --vf-muted: #6b7280;
    --vf-success: #1f9d55;
    --vf-warning: #d97706;
    --vf-danger: #dc2626;
    --radius-xl: 24px;
    --radius-lg: 18px;
    --radius-md: 14px;
}

.stApp {
    background: linear-gradient(180deg, #fff5f5 0%, #f7f8fa 22%, #f7f8fa 100%);
    color: var(--vf-text);
}

.block-container {
    max-width: 1300px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

.hero-box {
    background: linear-gradient(135deg, #fff5f5 0%, #ffffff 60%, #fff1f1 100%);
    border: 1px solid #ffd9d9;
    border-radius: var(--radius-xl);
    padding: 22px 24px;
    margin-bottom: 1rem;
    box-shadow: 0 12px 35px rgba(230, 0, 0, 0.08);
}

.main-title {
    font-size: 2.15rem;
    font-weight: 800;
    color: var(--vf-red);
    margin-bottom: .2rem;
    letter-spacing: -.02em;
}

.sub-title {
    color: var(--vf-muted);
    font-size: 1rem;
}

.section-card {
    background: var(--vf-card);
    border: 1px solid var(--vf-border);
    border-radius: var(--radius-lg);
    padding: 18px 18px 12px;
    margin-bottom: 16px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}

.metric-card {
    background: linear-gradient(180deg, #ffffff 0%, #fffafa 100%);
    border: 1px solid var(--vf-border);
    border-radius: var(--radius-lg);
    padding: 16px;
    min-height: 110px;
}

.metric-label {
    color: var(--vf-muted);
    font-size: .92rem;
    margin-bottom: 8px;
    font-weight: 600;
}

.metric-value {
    color: var(--vf-text);
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1.1;
}

.metric-sub {
    color: var(--vf-muted);
    font-size: .92rem;
    margin-top: 6px;
}

.band-low {
    border-left: 6px solid var(--vf-success);
}
.band-high {
    border-left: 6px solid var(--vf-warning);
}
.band-very-high {
    border-left: 6px solid var(--vf-danger);
}

.amount-box {
    background: #fff8f8;
    border: 1px dashed #f2b4b4;
    border-radius: 16px;
    padding: 16px;
    margin-top: 8px;
}

.small-note {
    color: var(--vf-muted);
    font-size: .92rem;
}

div[data-testid="stForm"] {
    background: var(--vf-card);
    border: 1px solid var(--vf-border);
    border-radius: 22px;
    padding: 16px 16px 8px 16px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--vf-border);
    border-radius: 16px;
    padding: 8px;
}

div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #d9dee8 !important;
    border-radius: 12px !important;
    color: var(--vf-text) !important;
    min-height: 46px !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    color: var(--vf-text) !important;
    border: 1px solid #d9dee8 !important;
    border-radius: 12px !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border: 1px solid var(--vf-red) !important;
    box-shadow: 0 0 0 2px rgba(230, 0, 0, 0.12) !important;
}

label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    color: var(--vf-text) !important;
    font-weight: 600 !important;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    background: linear-gradient(90deg, var(--vf-red), var(--vf-red-dark));
    color: white;
    font-weight: 700;
    padding: .72rem 1rem;
    box-shadow: 0 8px 18px rgba(230, 0, 0, 0.18);
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(90deg, #ff1a1a, var(--vf-red));
    transform: translateY(-1px);
}

div.stButton > button:focus,
div[data-testid="stFormSubmitButton"] > button:focus {
    box-shadow: 0 0 0 3px rgba(230, 0, 0, 0.16);
}

hr {
    border-color: #eef0f4;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--vf-border);
    border-radius: 16px;
    overflow: hidden;
    background: white;
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

h1, h2, h3 {
    color: var(--vf-text);
}

section[data-testid="stSidebar"] {
    background: #ffffff;
}
</style>
""", unsafe_allow_html=True)

def resolve_data_file():
    for file_name in POSSIBLE_DATA_FILES:
        if Path(file_name).exists():
            return file_name
    return None


@st.cache_data
def load_data():
    data_file = resolve_data_file()
    if data_file is None:
        st.error(
            "Dataset file was not found. Add 'Telco_customer_churn.xlsx' to your repo root "
            "or update POSSIBLE_DATA_FILES with the correct filename."
        )
        st.stop()
    return pd.read_excel(data_file), data_file


def clean_and_prepare(df):
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

    return X, y, num_cols, cat_cols


@st.cache_resource
def train_model():
    df, data_file = load_data()
    X, y, num_cols, cat_cols = clean_and_prepare(df)

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

    cat_options = {}
    for col in cat_cols:
        vals = X[col].dropna().astype(str).unique().tolist()
        vals = sorted([v for v in vals if v.lower() != "nan"])
        cat_options[col] = vals

    defaults = {}
    for col in X.columns:
        if col in num_cols:
            defaults[col] = float(pd.to_numeric(X[col], errors="coerce").median())
        else:
            mode_vals = X[col].mode()
            defaults[col] = str(mode_vals.iloc[0]) if not mode_vals.empty else ""

    return model, X.columns.tolist(), num_cols, cat_cols, cat_options, defaults, data_file


def pick_index(options, preferred):
    if preferred in options:
        return options.index(preferred)
    return 0


def risk_band(prob):
    if prob >= 0.75:
        return "Very High"
    elif prob >= 0.50:
        return "High"
    return "Low"


def band_class(prob):
    if prob >= 0.75:
        return "band-very-high"
    elif prob >= 0.50:
        return "band-high"
    return "band-low"


def safe_predict(input_dict):
    input_df = pd.DataFrame([input_dict]).reindex(columns=feature_order)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    return pred, prob, input_df


def simulate_charge_breakpoints(base_input, start_amt=0, end_amt=200, step=1, auto_total=True):
    tenure = max(float(base_input.get("Tenure Months", 0)), 1.0)
    rows = []

    for charge in np.arange(start_amt, end_amt + step, step):
        scenario = base_input.copy()
        scenario["Monthly Charges"] = round(float(charge), 2)

        if auto_total:
            scenario["Total Charges"] = round(float(charge) * tenure, 2)

        _, prob, _ = safe_predict(scenario)
        rows.append({
            "Monthly Charges": round(float(charge), 2),
            "Total Charges Used": round(float(scenario["Total Charges"]), 2),
            "Churn Probability": round(float(prob), 4),
            "Risk Band": risk_band(prob)
        })

    sim_df = pd.DataFrame(rows)

    high_row = sim_df[sim_df["Churn Probability"] >= 0.50].head(1)
    very_high_row = sim_df[sim_df["Churn Probability"] >= 0.75].head(1)

    high_amount = None if high_row.empty else float(high_row.iloc[0]["Monthly Charges"])
    very_high_amount = None if very_high_row.empty else float(very_high_row.iloc[0]["Monthly Charges"])

    return sim_df, high_amount, very_high_amount


model, feature_order, num_cols, cat_cols, cat_options, defaults, active_data_file = train_model()

st.markdown("""
<div class="hero-box">
    <div class="main-title">Customer Churn Prediction Dashboard</div>
    <div class="sub-title">
        Predict churn probability and estimate the monthly charge level where the customer may leave.
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(f"Loaded dataset: {active_data_file}")

left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Customer details")

    with st.form("churn_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            country = st.selectbox("Country", cat_options.get("Country", ["United States"]),
                                   index=pick_index(cat_options.get("Country", ["United States"]), defaults.get("Country", "United States")))
            state = st.text_input("State", defaults.get("State", "California"))
            city = st.text_input("City", defaults.get("City", "Los Angeles"))
            zip_code = st.number_input("Zip Code", min_value=1, value=int(defaults.get("Zip Code", 90001)))
            gender = st.selectbox("Gender", cat_options.get("Gender", ["Male", "Female"]),
                                  index=pick_index(cat_options.get("Gender", ["Male", "Female"]), defaults.get("Gender", "Male")))
            senior_citizen = st.selectbox("Senior Citizen", cat_options.get("Senior Citizen", ["Yes", "No"]),
                                          index=pick_index(cat_options.get("Senior Citizen", ["Yes", "No"]), defaults.get("Senior Citizen", "No")))
            partner = st.selectbox("Partner", cat_options.get("Partner", ["Yes", "No"]),
                                   index=pick_index(cat_options.get("Partner", ["Yes", "No"]), defaults.get("Partner", "No")))
            dependents = st.selectbox("Dependents", cat_options.get("Dependents", ["Yes", "No"]),
                                      index=pick_index(cat_options.get("Dependents", ["Yes", "No"]), defaults.get("Dependents", "No")))

        with c2:
            lat_long = st.text_input("Lat Long", defaults.get("Lat Long", "34.0522, -118.2437"))
            latitude = st.number_input("Latitude", value=float(defaults.get("Latitude", 34.0522)), format="%.6f")
            longitude = st.number_input("Longitude", value=float(defaults.get("Longitude", -118.2437)), format="%.6f")
            tenure_months = st.number_input("Tenure Months", min_value=0, max_value=120, value=int(defaults.get("Tenure Months", 12)))
            phone_service = st.selectbox("Phone Service", cat_options.get("Phone Service", ["Yes", "No"]),
                                         index=pick_index(cat_options.get("Phone Service", ["Yes", "No"]), defaults.get("Phone Service", "Yes")))
            multiple_lines = st.selectbox("Multiple Lines", cat_options.get("Multiple Lines", ["Yes", "No", "No phone service"]),
                                          index=pick_index(cat_options.get("Multiple Lines", ["Yes", "No", "No phone service"]), defaults.get("Multiple Lines", "No")))
            internet_service = st.selectbox("Internet Service", cat_options.get("Internet Service", ["DSL", "Fiber optic", "No"]),
                                            index=pick_index(cat_options.get("Internet Service", ["DSL", "Fiber optic", "No"]), defaults.get("Internet Service", "Fiber optic")))
            contract = st.selectbox("Contract", cat_options.get("Contract", ["Month-to-month", "One year", "Two year"]),
                                    index=pick_index(cat_options.get("Contract", ["Month-to-month", "One year", "Two year"]), defaults.get("Contract", "Month-to-month")))

        with c3:
            online_security = st.selectbox("Online Security", cat_options.get("Online Security", ["Yes", "No", "No internet service"]),
                                           index=pick_index(cat_options.get("Online Security", ["Yes", "No", "No internet service"]), defaults.get("Online Security", "No")))
            online_backup = st.selectbox("Online Backup", cat_options.get("Online Backup", ["Yes", "No", "No internet service"]),
                                         index=pick_index(cat_options.get("Online Backup", ["Yes", "No", "No internet service"]), defaults.get("Online Backup", "No")))
            device_protection = st.selectbox("Device Protection", cat_options.get("Device Protection", ["Yes", "No", "No internet service"]),
                                             index=pick_index(cat_options.get("Device Protection", ["Yes", "No", "No internet service"]), defaults.get("Device Protection", "No")))
            tech_support = st.selectbox("Tech Support", cat_options.get("Tech Support", ["Yes", "No", "No internet service"]),
                                        index=pick_index(cat_options.get("Tech Support", ["Yes", "No", "No internet service"]), defaults.get("Tech Support", "No")))
            streaming_tv = st.selectbox("Streaming TV", cat_options.get("Streaming TV", ["Yes", "No", "No internet service"]),
                                        index=pick_index(cat_options.get("Streaming TV", ["Yes", "No", "No internet service"]), defaults.get("Streaming TV", "No")))
            streaming_movies = st.selectbox("Streaming Movies", cat_options.get("Streaming Movies", ["Yes", "No", "No internet service"]),
                                            index=pick_index(cat_options.get("Streaming Movies", ["Yes", "No", "No internet service"]), defaults.get("Streaming Movies", "No")))
            paperless_billing = st.selectbox("Paperless Billing", cat_options.get("Paperless Billing", ["Yes", "No"]),
                                             index=pick_index(cat_options.get("Paperless Billing", ["Yes", "No"]), defaults.get("Paperless Billing", "Yes")))
            payment_method = st.selectbox("Payment Method", cat_options.get("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
                                          index=pick_index(cat_options.get("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]), defaults.get("Payment Method", "Electronic check")))

        st.markdown("---")
        b1, b2, b3 = st.columns(3)

        with b1:
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0,
                                              value=float(defaults.get("Monthly Charges", 70.0)), step=1.0)
        with b2:
            total_charges = st.number_input("Total Charges", min_value=0.0, max_value=20000.0,
                                            value=float(defaults.get("Total Charges", 1000.0)), step=10.0)
        with b3:
            auto_total = st.checkbox("Auto-estimate Total Charges", value=True)

        submitted = st.form_submit_button("Predict churn and analyze charge level")

    st.markdown('</div>', unsafe_allow_html=True)

base_input = {
    "Country": country,
    "State": state,
    "City": city,
    "Zip Code": zip_code,
    "Lat Long": lat_long,
    "Latitude": latitude,
    "Longitude": longitude,
    "Gender": gender,
    "Senior Citizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure_months,
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
    "Paperless Billing": paperless_billing,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": (monthly_charges * max(tenure_months, 1)) if auto_total else total_charges
}

if submitted:
    prediction, probability, input_df = safe_predict(base_input)
    sim_df, high_amount, very_high_amount = simulate_charge_breakpoints(base_input, 0, 200, 1, auto_total)

    with right:
        st.markdown(f"""
        <div class="section-card">
            <div class="metric-card {band_class(probability)}">
                <div class="metric-label">Prediction summary</div>
                <div class="metric-value">{'Customer may leave' if prediction == 1 else 'Customer may stay'}</div>
                <div class="metric-sub">Risk band: {risk_band(probability)} · Probability: {probability:.2%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        x1, x2 = st.columns(2)
        with x1:
            st.metric("Current monthly charges", f"${monthly_charges:,.2f}")
        with x2:
            st.metric("Total charges used", f"${base_input['Total Charges']:,.2f}")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Charge threshold insight")

        if high_amount is not None:
            st.markdown(f"""
            <div class="amount-box">
                <div class="metric-label">First amount where risk becomes High (50%+)</div>
                <div class="metric-value">${high_amount:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("No High-risk threshold found between $0 and $200.")

        if very_high_amount is not None:
            st.markdown(f"""
            <div class="amount-box">
                <div class="metric-label">First amount where risk becomes Very High (75%+)</div>
                <div class="metric-value">${very_high_amount:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No Very High-risk threshold found between $0 and $200.")

        st.markdown('<p class="small-note">This amount is profile-based, not a universal fixed rule.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Risk band")
    if probability >= 0.75:
        st.error("Very High risk")
    elif probability >= 0.50:
        st.warning("High risk")
    else:
        st.success("Low risk")

    st.subheader("Charge simulation")
    chart_df = sim_df.set_index("Monthly Charges")[["Churn Probability"]]
    st.line_chart(chart_df, height=320)

    show_df = sim_df.iloc[::10].copy()
    show_df["Churn Probability"] = show_df["Churn Probability"].map(lambda x: f"{x:.2%}")
    st.dataframe(show_df, use_container_width=True)

    st.subheader("Input summary")
    st.dataframe(input_df, use_container_width=True)
else:
    with right:
        st.markdown("""
        <div class="section-card">
            <div class="metric-card band-low">
                <div class="metric-label">What changed</div>
                <div class="metric-value">Better UI + safer file loading</div>
                <div class="metric-sub">This version checks multiple dataset names before loading.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
