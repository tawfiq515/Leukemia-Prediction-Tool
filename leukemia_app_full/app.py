
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import base64
import datetime
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from xgboost import XGBClassifier, XGBRegressor

# ========== Config ==========
st.set_page_config(page_title="Leukemia Predictor Tool", page_icon="🧪", layout="centered")

# --------- Customize ----------
BACKGROUND_IMG = "/home/tawfiq/Desktop/leukemia_app_full/midical.jpeg"
TEXT_COLOR     = "#0077B6"  # darker bold blue
# ------------------------------

def add_bg_from_local(image_file: str, text_color: str = "#000000"):
    """Set background image and force all text to a specific color & bold."""
    try:
        ext = image_file.split('.')[-1]
        mime_type = "jpeg" if ext == "jpg" else ext
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/{mime_type};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        /* Force text color & bold */
        html, body, .stApp, * {{
            color: {text_color} !important;
            font-weight: bold !important;
        }}
        /* Inputs styling */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
        .stFileUploader div[role="button"], .stButton button, .stRadio, .stMarkdown, .stDataFrame {{
            background: rgba(255,255,255,0.08) !important;
            border-radius: 8px !important;
        }}
        .stButton button {{
            border: 1px solid {text_color} !important;
            font-weight: bold !important;
        }}
        section[data-testid="stSidebar"] * {{
            color: {text_color} !important;
            font-weight: bold !important;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Background image could not be loaded: {e}")

# ========== Defaults & Aliases ==========
NORMAL_VALUES = {
    "WBC": 7.0,
    "RBC": 4.5,
    "Hb": 13.5,
    "Platelets": 250.0,
    "Lymphocytes": 30.0,
}
FEATURES = list(NORMAL_VALUES.keys())

ALIAS_MAP = {
    "wbc": "WBC", "white blood cells": "WBC", "white_blood_cells": "WBC",
    "total wbc": "WBC", "w.b.c": "WBC",
    "rbc": "RBC", "red blood cells": "RBC", "red_blood_cells": "RBC", "r.b.c": "RBC",
    "hb": "Hb", "hgb": "Hb", "hemoglobin": "Hb", "haemoglobin": "Hb",
    "platelets": "Platelets", "plt": "Platelets", "platelet": "Platelets",
    "lymphocytes": "Lymphocytes", "lymphs": "Lymphocytes", "lym": "Lymphocytes",
}

# ========== Helpers ==========
def normalize_name(name: str) -> str:
    n = str(name).strip().lower()
    n = n.replace("-", " ").replace("_", " ").replace(".", " ")
    n = re.sub(r"\s+", " ", n)
    return n

def unify_columns(df: pd.DataFrame):
    """Map CSV columns to standard names, create missing with defaults, coerce to numeric."""
    rename_map = {}
    for col in df.columns:
        norm = normalize_name(col)
        if norm in ALIAS_MAP:
            rename_map[col] = ALIAS_MAP[norm]
    df = df.rename(columns=rename_map)

    created_cols = []
    for col in FEATURES:
        if col not in df.columns:
            df[col] = NORMAL_VALUES[col]
            created_cols.append(col)

    df = df[[c for c in FEATURES]]
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(NORMAL_VALUES[col], inplace=True)

    detected_cols = [c for c in FEATURES if c not in created_cols]
    return df, detected_cols, created_cols

@st.cache_data(show_spinner=False)
def load_models(random_state: int = 42):
    np.random.seed(random_state)
    X_train = pd.DataFrame({
        "WBC":         np.random.normal(7,   3, 200),
        "RBC":         np.random.normal(4.5, 0.5, 200),
        "Hb":          np.random.normal(13.5, 2, 200),
        "Platelets":   np.random.normal(250, 50, 200),
        "Lymphocytes": np.random.normal(30, 10, 200),
    })
    y_class = (X_train["WBC"] > 10).astype(int)
    y_reg   = (X_train["WBC"] / 20)

    models = {
        "classification": {
            "Random Forest": RandomForestClassifier(random_state=random_state).fit(X_train, y_class),
            "XGBoost": XGBClassifier(
                eval_metric="logloss", random_state=random_state,
                n_estimators=150, max_depth=4
            ).fit(X_train, y_class),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state).fit(X_train, y_class),
        },
        "regression": {
            "Random Forest Regressor": RandomForestRegressor(random_state=random_state).fit(X_train, y_reg),
            "XGBoost Regressor": XGBRegressor(
                n_estimators=200, max_depth=4, random_state=random_state
            ).fit(X_train, y_reg),
            "Linear Regression": LinearRegression().fit(X_train, y_reg),
        },
        "clustering": {
            "KMeans": KMeans(n_clusters=2, n_init=10, random_state=random_state),
            "DBSCAN": DBSCAN(eps=3),
            "Agglomerative": AgglomerativeClustering(n_clusters=2),
        }
    }
    return models

def save_user_analysis(username: str, features_row: dict, risk_score: float):
    filename = f"history_{username}.csv"
    row = features_row.copy()
    row["Risk"] = float(risk_score)
    row["Date"] = datetime.datetime.now().strftime("%Y-%m-%d")
    df_new = pd.DataFrame([row])
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(filename, index=False)

def compare_with_previous(username: str, current_risk: float) -> str:
    filename = f"history_{username}.csv"
    if not os.path.exists(filename):
        return "First time analysis. No previous results."
    df = pd.read_csv(filename)
    if len(df) < 2:
        return "Not enough history for comparison."
    previous_risk = float(df.iloc[-2]["Risk"])
    diff = current_risk - previous_risk
    if diff > 20:
        return f"Risk increased (+{diff:.2f}%). Please consult a doctor."
    elif diff < -20:
        return f"Risk decreased (−{abs(diff):.2f}%). Keep it up!"
    else:
        return f"Risk is stable with change of {diff:.2f}%."

# ✅ Detailed Recommendations
def give_recommendation(value, normal, feature):
    if value < 0.9 * normal:
        return f"⚠️ {feature} is below normal. Eat more iron-rich foods."
    elif value > 1.1 * normal:
        return f"❗ {feature} is above normal. Avoid processed foods and consult doctor."
    else:
        return f"✅ {feature} is within normal range."

# ========== App ==========
def main():
    # Background + colored text
    add_bg_from_local(BACKGROUND_IMG, TEXT_COLOR)

    st.title("Leukemia Prediction Tool (Robust CSV Support)")

    st.sidebar.title("User Login")
    username = st.sidebar.text_input("Enter your name:")
    if not username:
        st.warning("Please enter your name to continue.")
        st.stop()
    login_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.success(f"Logged in at {login_time}")

    if st.sidebar.button("Show My Previous Results"):
        fname = f"history_{username}.csv"
        if os.path.exists(fname):
            st.subheader("Previous Results")
            st.dataframe(pd.read_csv(fname))
        else:
            st.info("No previous records found.")

    models = load_models()

    st.markdown(
        """
        **Instructions**
        - Upload a CSV with any column names; the app will auto-map to standard features.
        - Required standard features: WBC, RBC, Hb, Platelets, Lymphocytes.
        - Missing features will be auto-created using NORMAL_VALUES.
        """
    )

    input_method = st.radio("Choose input method:", ("Upload CSV", "Manual Input"))

    if input_method == "Manual Input":
        data = {f: [st.number_input(f, value=float(NORMAL_VALUES[f]))] for f in FEATURES}
        df_input = pd.DataFrame(data)
        detected_cols, created_cols = FEATURES, []
    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if not file:
            st.stop()
        raw_df = pd.read_csv(file)
        df_input, detected_cols, created_cols = unify_columns(raw_df)

        if detected_cols:
            st.success(f"Detected columns: {', '.join(detected_cols)}")
        if created_cols:
            st.warning(
                "These columns were missing and auto-filled with default normal values: "
                + ", ".join(created_cols)
            )

        with st.expander("Preview normalized input dataframe"):
            st.dataframe(df_input)

    if st.button("Analyze Current Test"):
        st.subheader("Classification Results")
        for name, model in models["classification"].items():
            pred = model.predict(df_input)[0]
            label = "Leukemia" if int(pred) == 1 else "Healthy"
            st.write(f"{name}: {label}")

        st.subheader("Risk Prediction (Regression)")
        reg_risks = {}
        for name, model in models["regression"].items():
            risk_score = float(np.clip(model.predict(df_input)[0], 0, 1) * 100)
            reg_risks[name] = risk_score
            st.write(f"{name}: {risk_score:.2f}%")

        st.subheader("Recommendations")
        for feature in FEATURES:
            rec = give_recommendation(float(df_input[feature][0]), NORMAL_VALUES[feature], feature)
            st.write(f"- {rec}")

        chosen_risk = reg_risks.get("Random Forest Regressor", list(reg_risks.values())[0])
        save_user_analysis(username, df_input.iloc[0].to_dict(), chosen_risk)
        st.info(compare_with_previous(username, chosen_risk))

        st.subheader("Clustering (for reference)")
        for name, model in models["clustering"].items():
            try:
                if hasattr(model, "fit_predict"):
                    cluster = model.fit_predict(df_input)[0]
                else:
                    model.fit(df_input)
                    cluster = getattr(model, "labels_", ["N/A"])[0]
            except Exception:
                cluster = "N/A"
            st.write(f"{name}: Cluster {cluster}")

if __name__ == "__main__":
    main()
