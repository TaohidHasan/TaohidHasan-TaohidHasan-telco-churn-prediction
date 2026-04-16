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

DATA_FILE = "Telcocustomerchurn.xlsx"

st.markdown("""
<style>
:root {
    --bg: #0f172a;
    --card: #111827;
    --card-2: #1f2937;
    --soft: #334155;
    --text: #f8fafc;
    --muted: #cbd5e1;
    --accent: #14b8a6;
    --accent-2: #0ea5e9;
    --success: #16a34a;
    --warning: #f59e0b;
    --danger: #ef4444;
    --radius: 18px;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #0b1120 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1300px;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.25rem;
}

.sub-title {
    color: #cbd5e1;
    font-size: 1rem;
    margin-bottom: 1.25rem;
}

.hero-box {
    background: linear-gradient(135deg, rgba(20,184,166,.18), rgba(14,165,233,.10));
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 24px;
    padding: 22px 24px;
    margin-bottom: 1rem;
    box-shadow: 0 10px 35px rgba(0,0,0,.25);
}

.section-card {
    background: rgba(17,24,39,.78);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: var(--radius);
    padding: 18px 18px 10px 18px;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,.22);
    backdrop-filter: blur(10px);
}

.metric-card {
    background: linear-gradient(180deg, rgba(31,41,55,.95), rgba(17,24,39,.95));
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px;
    padding: 16px;
    min-height: 110px;
}

.metric-label {
    color: #94a3b8;
    font-size: 0.92rem;
    margin-bottom: 8px;
}

.metric-value {
    color: white;
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1.1;
}

.metric-sub {
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-top: 6px;
}

.band-low {
    border-left: 6px solid #16a34a;
}
.band-high {
    border-left: 6px solid #f59e0b;
}
.band-very-high {
    border-left: 6px solid #ef4444;
}

.small-note {
    color: #cbd5e1;
    font-size: 0.92rem;
}

.amount-box {
    background: rgba(15,23,42,.9);
    border: 1px dashed rgba(20,184,166,.55);
    border-radius: 18px;
    padding: 16px;
    margin-top: 8px;
}

div[data-testid="stForm"] {
    background: rgba(17,24,39,.72);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 22px;
    padding: 16px 16px 8px 16px;
}

div[data-testid="stMetric"] {
    background: rgba(17,24,39,.6);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px;
    padding: 8px;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    background: linear-gradient(90deg, #14b8a6, #0ea5e9);
    color: white;
    font-weight: 700;
    padding: 0.7rem 1rem;
    box-shadow: 0 8px 20px rgba(20,184,166,.25);
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
}

div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    border-radius: 12px !important;
}

hr {
    border-color: rgba(255,255,255,.08);
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_excel(DATA_FILE)


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
    df = load_data()
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

    return model, X.columns.tolist(), num_cols, cat_cols, cat_options, defaults


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


model, feature_order, num_cols, cat_cols, cat_options, defaults = train_model()

st.markdown("""
<div class="hero-box">
    <div class="main-title">Customer Churn Prediction Dashboard</div>
    <div class="sub-title">
        Fill in the customer profile, predict churn probability, and estimate the monthly charge level
        where the customer is likely to leave.
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Customer details")

    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            country = st.selectbox(
                "Country",
                cat_options.get("Country", ["United States"]),
                index=pick_index(cat_options.get("Country", ["United States"]), defaults.get("Country", "United States"))
            )
            state = st.text_input("State", defaults.get("State", "California"))
            city = st.text_input("City", defaults.get("City", "Los Angeles"))
            zip_code = st.number_input("Zip Code", min_value=1, value=int(defaults.get("Zip Code", 90001)))

            gender = st.selectbox(
                "Gender",
                cat_options.get("Gender", ["Male", "Female"]),
                index=pick_index(cat_options.get("Gender", ["Male", "Female"]), defaults.get("Gender", "Male"))
            )
            senior_citizen = st.selectbox(
                "Senior Citizen",
                cat_options.get("Senior Citizen", ["Yes", "No"]),
                index=pick_index(cat_options.get("Senior Citizen", ["Yes", "No"]), defaults.get("Senior Citizen", "No"))
            )
            partner = st.selectbox(
                "Partner",
                cat_options.get("Partner", ["Yes", "No"]),
                index=pick_index(cat_options.get("Partner", ["Yes", "No"]), defaults.get("Partner", "No"))
            )
            dependents = st.selectbox(
                "Dependents",
                cat_options.get("Dependents", ["Yes", "No"]),
                index=pick_index(cat_options.get("Dependents", ["Yes", "No"]), defaults.get("Dependents", "No"))
            )

        with col2:
            lat_long = st.text_input("Lat Long", defaults.get("Lat Long", "34.0522, -118.2437"))
            latitude = st.number_input("Latitude", value=float(defaults.get("Latitude", 34.0522)), format="%.6f")
            longitude = st.number_input("Longitude", value=float(defaults.get("Longitude", -118.2437)), format="%.6f")
            tenure_months = st.number_input("Tenure Months", min_value=0, max_value=120, value=int(defaults.get("Tenure Months", 12)))

            phone_service = st.selectbox(
                "Phone Service",
                cat_options.get("Phone Service", ["Yes", "No"]),
                index=pick_index(cat_options.get("Phone Service", ["Yes", "No"]), defaults.get("Phone Service", "Yes"))
            )
            multiple_lines = st.selectbox(
                "Multiple Lines",
                cat_options.get("Multiple Lines", ["Yes", "No", "No phone service"]),
                index=pick_index(cat_options.get("Multiple Lines", ["Yes", "No", "No phone service"]), defaults.get("Multiple Lines", "No"))
            )
            internet_service = st.selectbox(
                "Internet Service",
                cat_options.get("Internet Service", ["DSL", "Fiber optic", "No"]),
                index=pick_index(cat_options.get("Internet Service", ["DSL", "Fiber optic", "No"]), defaults.get("Internet Service", "Fiber optic"))
            )
            contract = st.selectbox(
                "Contract",
                cat_options.get("Contract", ["Month-to-month", "One year", "Two year"]),
                index=pick_index(cat_options.get("Contract", ["Month-to-month", "One year", "Two year"]), defaults.get("Contract", "Month-to-month"))
            )

        with col3:
            online_security = st.selectbox(
                "Online Security",
                cat_options.get("Online Security", ["Yes", "No", "No internet service"]),
                index=pick_index(cat_options.get("Online Security", ["Yes", "No", "No internet service"]), defaults.get("Online Security", "No"))
            )
            online_backup = st.selectbox(
                "Online Backup",
                cat_options.get("Online Backup", ["Yes", "No", "No internet service"]),
                index=pick_index(cat_options.get("Online Backup", ["Yes", "No", "No internet service"]), defaults.get("Online Backup", "No"))
            )
            device_protection = st.selectbox(
                "Device Protection",
                cat_options.get("Device Protection", ["Yes", "No", "No internet service"]),
                index=pick_index(cat_options.get("Device Protection", ["Yes", "No", "No internet service"]), defaults.get("Device Protection", "No"))
            )
            tech_support = st.selectbox(
                "Tech Support",
                cat_options.get("Tech Support", ["Yes", "No", "No internet service"]),
                index=pick_index(cat_options.get("Tech Support", ["Yes", "No", "No internet service"]), defaults.get("Tech Support", "No"))
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                cat_options.get("Streaming TV", ["Yes", "No", "No internet service"]),
                index=pick_index(cat_options.get("Streaming TV", ["Yes", "No", "No internet service"]), defaults.get("Streaming TV", "No"))
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                cat_options.get("Streaming Movies", ["Yes", "No", "No internet service"]),
                index=pick_index(cat_options.get("Streaming Movies", ["Yes", "No", "No internet service"]), defaults.get("Streaming Movies", "No"))
            )
            paperless_billing = st.selectbox(
                "Paperless Billing",
                cat_options.get("Paperless Billing", ["Yes", "No"]),
                index=pick_index(cat_options.get("Paperless Billing", ["Yes", "No"]), defaults.get("Paperless Billing", "Yes"))
            )
            payment_method = st.selectbox(
                "Payment Method",
                cat_options.get("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
                index=pick_index(
                    cat_options.get("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
                    defaults.get("Payment Method", "Electronic check")
                )
            )

        st.markdown("---")
        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=float(defaults.get("Monthly Charges", 70.0)),
                step=1.0
            )

        with c2:
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=20000.0,
                value=float(defaults.get("Total Charges", 1000.0)),
                step=10.0
            )

        with c3:
            auto_total = st.checkbox(
                "Auto-estimate Total Charges from tenure × monthly charges",
                value=True
            )

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
    sim_df, high_amount, very_high_amount = simulate_charge_breakpoints(
        base_input,
        start_amt=0,
        end_amt=200,
        step=1,
        auto_total=auto_total
    )

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

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Current monthly charges", f"${monthly_charges:,.2f}")
        with c2:
            st.metric("Total charges used", f"${base_input['Total Charges']:,.2f}")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Charge threshold insight")

        if high_amount is not None:
            st.markdown(
                f"""
                <div class="amount-box">
                    <div class="metric-label">First charge where risk becomes <b>High</b> (50%+)</div>
                    <div class="metric-value">${high_amount:,.2f}</div>
                    <div class="metric-sub">This is the first monthly charge in the simulator range where churn probability reaches at least 50%.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.success("No High-risk threshold found between $0 and $200 for this customer profile.")

        if very_high_amount is not None:
            st.markdown(
                f"""
                <div class="amount-box">
                    <div class="metric-label">First charge where risk becomes <b>Very High</b> (75%+)</div>
                    <div class="metric-value">${very_high_amount:,.2f}</div>
                    <div class="metric-sub">This is the first monthly charge in the simulator range where churn probability reaches at least 75%.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No Very High-risk threshold found between $0 and $200 for this customer profile.")

        st.markdown(
            '<p class="small-note">Tip: this threshold is profile-based, so it changes with tenure, contract, internet service, payment method, and other customer details.</p>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Risk band")
    if probability >= 0.75:
        st.error("Very High risk: the selected customer profile is strongly likely to churn.")
    elif probability >= 0.50:
        st.warning("High risk: the customer profile is leaning toward churn.")
    else:
        st.success("Low risk: the customer profile is more likely to stay.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Why this result may happen")

    notes = []
    if monthly_charges > 80:
        notes.append("Higher monthly charges can increase price pressure.")
    if contract == "Month-to-month":
        notes.append("Month-to-month contracts often increase churn sensitivity.")
    if tenure_months < 12:
        notes.append("Short tenure means the customer is still early in the relationship.")
    if payment_method == "Electronic check":
        notes.append("Electronic check can be associated with higher churn sensitivity.")
    if internet_service == "Fiber optic":
        notes.append("Fiber optic customers may show stronger churn variation depending on price and service expectations.")
    if not notes:
        notes.append("No major warning signal was triggered by the simple rule-based notes, but the final probability still comes from the ML model.")

    for note in notes:
        st.write(f"- {note}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Charge simulation chart")

    chart_df = sim_df.copy()
    chart_df = chart_df.set_index("Monthly Charges")[["Churn Probability"]]
    st.line_chart(chart_df, height=320)

    show_df = sim_df.iloc[::10].copy()
    show_df["Churn Probability"] = show_df["Churn Probability"].map(lambda x: f"{x:.2%}")
    st.dataframe(show_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Input summary")
    st.dataframe(input_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    with right:
        st.markdown("""
        <div class="section-card">
            <div class="metric-card band-low">
                <div class="metric-label">What this improved app does</div>
                <div class="metric-value">Cleaner UI</div>
                <div class="metric-sub">
                    Probability-based bands, better layout, charge simulator, and easier customer input flow.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
            <div class="metric-label">How to use</div>
            <div class="metric-sub">
                1. Fill in the customer profile.<br>
                2. Click the prediction button.<br>
                3. Check the charge threshold boxes to see from which monthly amount the customer may become high risk.
            </div>
        </div>
        """, unsafe_allow_html=True)
