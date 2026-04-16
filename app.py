from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

DATA_CANDIDATES = [
    Path("Telco_customer_churn.xlsx"),
    Path("Telcocustomerchurn.xlsx"),
    Path("telco_customer_churn.xlsx"),
]
MODEL_CANDIDATES = [
    Path("bestpipeline.pkl"),
    Path("best_pipeline.pkl"),
]
TARGET = "Churn Value"
DROP_COLS = ["CustomerID", "Count", "Churn Label", "Churn Score", "CLTV", "Churn Reason"]
NUMERIC_FORCE = ["Zip Code", "Latitude", "Longitude", "Tenure Months", "Monthly Charges", "Total Charges"]


def find_first(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def clean_features(df: pd.DataFrame):
    x = df.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df.columns]).copy()
    x = x.replace(["", " ", "NA", "N/A", "null", "None"], np.nan)

    numeric_force = [c for c in NUMERIC_FORCE if c in x.columns]
    for col in numeric_force:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    num_cols = x.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = [c for c in x.columns if c not in num_cols]
    for col in cat_cols:
        x[col] = x[col].astype(str)
    return x, num_cols, cat_cols


@st.cache_resource
def load_artifacts():
    data_path = find_first(DATA_CANDIDATES)
    model_path = find_first(MODEL_CANDIDATES)
    if data_path is None:
        raise FileNotFoundError("Dataset file not found. Place Telco_customer_churn.xlsx or Telcocustomerchurn.xlsx beside app.py.")

    df = pd.read_excel(data_path)
    x, num_cols, cat_cols = clean_features(df)
    y = df[TARGET].copy()

    cat_options = {}
    for col in cat_cols:
        vals = sorted(v for v in x[col].dropna().astype(str).unique().tolist() if v != "nan")
        cat_options[col] = vals

    numeric_defaults = {}
    for col in num_cols:
        series = pd.to_numeric(x[col], errors="coerce")
        numeric_defaults[col] = float(series.median()) if not series.dropna().empty else 0.0

    if model_path is not None:
        model = joblib.load(model_path)
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), num_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), cat_cols),
            ]
        )
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(random_state=42)),
        ])
        model.fit(x, y)

    return {
        "df": df,
        "model": model,
        "features": x.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_options": cat_options,
        "numeric_defaults": numeric_defaults,
        "data_path": str(data_path),
        "model_path": str(model_path) if model_path else None,
    }


def get_numeric_input(col, default):
    integer_like = col in {"Zip Code", "Tenure Months"}
    if integer_like:
        if col == "Zip Code":
            return int(st.number_input(col, min_value=0, max_value=99999, value=int(round(default or 0)), step=1))
        return int(st.number_input(col, min_value=0, max_value=120, value=int(round(default or 0)), step=1))
    if col == "Longitude":
        return float(st.number_input(col, value=float(default or 0.0), format="%.6f"))
    if col == "Latitude":
        return float(st.number_input(col, value=float(default or 0.0), format="%.6f"))
    if col == "Monthly Charges":
        return float(st.number_input(col, min_value=0.0, max_value=500.0, value=float(default or 0.0), step=0.1))
    if col == "Total Charges":
        return float(st.number_input(col, min_value=0.0, max_value=50000.0, value=float(default or 0.0), step=0.1))
    return float(st.number_input(col, value=float(default or 0.0), step=0.1))


def pick_default(options, preferred):
    for item in preferred:
        if item in options:
            return options.index(item)
    return 0


def get_categorical_input(col, options):
    preferred_map = {
        "Country": ["United States"],
        "State": ["California"],
        "Gender": ["Male", "Female"],
        "Senior Citizen": ["No", "Yes"],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "Phone Service": ["Yes", "No"],
        "Multiple Lines": ["No", "Yes", "No phone service"],
        "Internet Service": ["Fiber optic", "DSL", "No"],
        "Online Security": ["No", "Yes", "No internet service"],
        "Online Backup": ["No", "Yes", "No internet service"],
        "Device Protection": ["No", "Yes", "No internet service"],
        "Tech Support": ["No", "Yes", "No internet service"],
        "Streaming TV": ["No", "Yes", "No internet service"],
        "Streaming Movies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Paperless Billing": ["Yes", "No"],
        "Payment Method": ["Electronic check", "Mailed check", "Bank transfer automatic", "Credit card automatic"],
    }

    if not options:
        return st.text_input(col, "")

    if len(options) <= 6:
        idx = pick_default(options, preferred_map.get(col, []))
        return st.selectbox(col, options, index=idx)

    if col in {"City", "State", "Country"}:
        default = options[pick_default(options, preferred_map.get(col, []))]
        return st.selectbox(col, options, index=options.index(default))

    return st.selectbox(col, options, index=0)


def risk_label(prob):
    if prob >= 0.70:
        return "High risk", "error"
    if prob >= 0.40:
        return "Medium risk", "warning"
    return "Low risk", "success"


art = load_artifacts()
model = art["model"]
features = art["features"]
num_cols = art["num_cols"]
cat_cols = art["cat_cols"]
cat_options = art["cat_options"]
numeric_defaults = art["numeric_defaults"]

st.title("📉 Customer Churn Prediction")
st.caption("Predict whether a telecom customer is likely to churn using the trained Telco churn model.")

with st.sidebar:
    st.subheader("Model assets")
    st.write(f"Dataset: `{art['data_path']}`")
    st.write(f"Pipeline: `{art['model_path'] or 'Trained automatically from dataset'}`")
    st.subheader("How to use")
    st.markdown(
        "1. Enter the customer details.\n"
        "2. Click **Predict churn**.\n"
        "3. Review the probability, risk band, and recommendation."
    )

left, right = st.columns([1.15, 0.85])

with left:
    with st.form("churn_form"):
        st.subheader("Customer details")

        groups = {
            "Profile": ["Country", "State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude", "Gender", "Senior Citizen", "Partner", "Dependents"],
            "Services": ["Phone Service", "Multiple Lines", "Internet Service", "Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"],
            "Billing": ["Tenure Months", "Contract", "Paperless Billing", "Payment Method", "Monthly Charges", "Total Charges"],
        }

        input_data = {}
        rendered = set()

        for section, cols in groups.items():
            st.markdown(f"### {section}")
            c1, c2 = st.columns(2)
            slots = [c1, c2]
            i = 0
            for col in cols:
                if col not in features:
                    continue
                with slots[i % 2]:
                    if col in num_cols:
                        input_data[col] = get_numeric_input(col, numeric_defaults.get(col, 0.0))
                    elif col == "Lat Long":
                        lat = numeric_defaults.get("Latitude", 0.0)
                        lon = numeric_defaults.get("Longitude", 0.0)
                        input_data[col] = st.text_input(col, f"{lat:.6f}, {lon:.6f}")
                    else:
                        input_data[col] = get_categorical_input(col, cat_options.get(col, []))
                rendered.add(col)
                i += 1

        remaining = [c for c in features if c not in rendered]
        if remaining:
            st.markdown("### Additional fields")
            c1, c2 = st.columns(2)
            slots = [c1, c2]
            for i, col in enumerate(remaining):
                with slots[i % 2]:
                    if col in num_cols:
                        input_data[col] = get_numeric_input(col, numeric_defaults.get(col, 0.0))
                    else:
                        input_data[col] = get_categorical_input(col, cat_options.get(col, []))

        submitted = st.form_submit_button("Predict churn", use_container_width=True)

with right:
    st.subheader("Project overview")
    st.markdown(
        "This app accepts customer service, contract, billing, and demographic information and returns a churn prediction with an estimated probability score."
    )
    st.metric("Rows in dataset", f"{len(art['df']):,}")
    st.metric("Input features", len(features))
    st.metric("Categorical features", len(cat_cols))

    churn_rate = float(art["df"][TARGET].mean()) if TARGET in art["df"].columns else 0.0
    st.metric("Observed churn rate", f"{churn_rate:.1%}")

if submitted:
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=features)

    pred = int(model.predict(input_df)[0])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_df)[0][1])
    else:
        score = float(model.decision_function(input_df)[0])
        prob = 1 / (1 + np.exp(-score))

    label, tone = risk_label(prob)

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Predicted churn", "Yes" if pred == 1 else "No")
    b.metric("Churn probability", f"{prob:.2%}")
    c.metric("Risk band", label)

    message = {
        "error": st.error,
        "warning": st.warning,
        "success": st.success,
    }[tone]
    message(f"{label}: the customer shows an estimated churn probability of {prob:.2%}.")

    top_signals = []
    if input_data.get("Contract") == "Month-to-month":
        top_signals.append("Month-to-month contracts often align with higher churn risk.")
    if input_data.get("Internet Service") == "Fiber optic":
        top_signals.append("Fiber optic customers can show higher churn in this dataset.")
    if float(input_data.get("Tenure Months", 0) or 0) < 12:
        top_signals.append("Short tenure may indicate a less stable customer relationship.")
    if input_data.get("Payment Method") == "Electronic check":
        top_signals.append("Electronic check appears frequently among higher-risk cases.")
    if float(input_data.get("Monthly Charges", 0) or 0) > 80:
        top_signals.append("Higher monthly charges may increase churn pressure.")

    st.subheader("Prediction notes")
    if top_signals:
        for item in top_signals[:4]:
            st.write(f"- {item}")
    else:
        st.write("- No obvious high-risk pattern was triggered by the entered values.")

    st.subheader("Entered data")
    st.dataframe(input_df, use_container_width=True)
