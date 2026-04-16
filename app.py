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
        ("preprocessor", 
