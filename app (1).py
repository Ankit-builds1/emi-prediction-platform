import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import os
import sqlite3
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="EMI Prediction Platform", layout="wide")

# ============================================================
# DAGSHUB / MLFLOW CONNECTION
# ============================================================
# Reads from Streamlit secrets (Streamlit Cloud) if available,
# otherwise falls back to OS environment variables (Render, HF, etc.)
def get_config(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key)

DAGSHUB_USERNAME = get_config("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = get_config("DAGSHUB_TOKEN")
DAGSHUB_REPO = get_config("DAGSHUB_REPO")  # e.g. "emi-prediction-platform"

if not all([DAGSHUB_USERNAME, DAGSHUB_TOKEN, DAGSHUB_REPO]):
    st.error("Missing DagsHub configuration. Please set DAGSHUB_USERNAME, "
             "DAGSHUB_TOKEN, and DAGSHUB_REPO as secrets or environment variables.")
    st.stop()

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")


@st.cache_resource
def load_models():
    """Load the production classifier and regressor from the MLflow registry."""
    clf = mlflow.pyfunc.load_model(f"models:/EMI_Eligibility_Classifier/Production")
    reg = mlflow.pyfunc.load_model(f"models:/Max_EMI_Regressor/Production")
    return clf, reg


# ============================================================
# LOCAL SQLITE DB FOR CRUD (financial records)
# ============================================================
DB_PATH = "emi_records.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS applicants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            monthly_salary REAL,
            credit_score REAL,
            requested_amount REAL,
            predicted_eligibility TEXT,
            predicted_max_emi REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def add_record(name, age, salary, credit_score, requested_amount, eligibility, max_emi):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO applicants
           (name, age, monthly_salary, credit_score, requested_amount,
            predicted_eligibility, predicted_max_emi, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (name, age, salary, credit_score, requested_amount, eligibility, max_emi,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


def get_records():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM applicants ORDER BY id DESC", conn)
    conn.close()
    return df


def update_record(record_id, name, age, salary, credit_score, requested_amount):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """UPDATE applicants
           SET name=?, age=?, monthly_salary=?, credit_score=?, requested_amount=?
           WHERE id=?""",
        (name, age, salary, credit_score, requested_amount, record_id),
    )
    conn.commit()
    conn.close()


def delete_record(record_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM applicants WHERE id=?", (record_id,))
    conn.commit()
    conn.close()


init_db()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("EMI Prediction Platform")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict EMI Eligibility", "Data Explorer", "Manage Records (CRUD)", "Model Dashboard"],
)

# ============================================================
# PAGE: HOME
# ============================================================
if page == "Home":
    st.title("EMI Prediction Platform")
    st.markdown("""
    Welcome. This platform predicts:
    - **EMI Eligibility** (Eligible / High Risk / Not Eligible)
    - **Maximum safe monthly EMI** amount

    Use the sidebar to navigate between prediction, data exploration,
    record management, and model performance dashboards.
    """)
    st.info("Models are loaded live from the MLflow Model Registry hosted on DagsHub.")

# ============================================================
# PAGE: PREDICTION
# ============================================================
elif page == "Predict EMI Eligibility":
    st.title("Predict EMI Eligibility & Max EMI")

    with st.spinner("Loading models from MLflow registry..."):
        clf_model, reg_model = load_models()

    st.subheader("Applicant Details")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Applicant Name")
        age = st.number_input("Age", min_value=18, max_value=75, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox("Education", ["Graduate", "Post Graduate", "Undergraduate", "PhD"])
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business"])
        company_type = st.selectbox("Company Type", ["Government", "Private", "MNC", "Startup"])
        monthly_salary = st.number_input("Monthly Salary (INR)", min_value=0.0, value=50000.0)
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=5.0)

    with col2:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
        bank_balance = st.number_input("Bank Balance (INR)", min_value=0.0, value=100000.0)
        emergency_fund = st.number_input("Emergency Fund (INR)", min_value=0.0, value=50000.0)
        existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
        current_emi_amount = st.number_input("Current EMI Amount (INR)", min_value=0.0, value=0.0)
        house_type = st.selectbox("House Type", ["Owned", "Rented", "Family Owned"])
        monthly_rent = st.number_input("Monthly Rent (INR)", min_value=0.0, value=0.0)
        family_size = st.number_input("Family Size", min_value=1, value=3)
        dependents = st.number_input("Dependents", min_value=0, value=1)

    st.subheader("Expenses")
    col3, col4 = st.columns(2)
    with col3:
        school_fees = st.number_input("School Fees (INR)", min_value=0.0, value=0.0)
        college_fees = st.number_input("College Fees (INR)", min_value=0.0, value=0.0)
        travel_expenses = st.number_input("Travel Expenses (INR)", min_value=0.0, value=3000.0)
    with col4:
        groceries_utilities = st.number_input("Groceries & Utilities (INR)", min_value=0.0, value=8000.0)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (INR)", min_value=0.0, value=2000.0)

    st.subheader("Loan Request")
    col5, col6 = st.columns(2)
    with col5:
        requested_amount = st.number_input("Requested Loan Amount (INR)", min_value=0.0, value=200000.0)
    with col6:
        requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, value=24)
        emi_scenario = st.selectbox("EMI Scenario", ["New Loan", "Top Up", "Balance Transfer"])

    if st.button("Predict", type="primary"):
        # NOTE: Replace this block with your actual scaler + label encoders
        # loaded via joblib if you want exact parity with training preprocessing.
        # This is a simplified structure matching your training feature set.
        input_dict = {
            "age": age, "gender": gender, "marital_status": marital_status,
            "education": education, "employment_type": employment_type,
            "company_type": company_type, "monthly_salary": monthly_salary,
            "years_of_employment": years_of_employment, "credit_score": credit_score,
            "bank_balance": bank_balance, "emergency_fund": emergency_fund,
            "existing_loans": existing_loans, "current_emi_amount": current_emi_amount,
            "house_type": house_type, "monthly_rent": monthly_rent,
            "family_size": family_size, "dependents": dependents,
            "school_fees": school_fees, "college_fees": college_fees,
            "travel_expenses": travel_expenses, "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "requested_amount": requested_amount, "requested_tenure": requested_tenure,
            "emi_scenario": emi_scenario,
        }
        input_df = pd.DataFrame([input_dict])

        try:
            eligibility_pred = clf_model.predict(input_df)[0]
            max_emi_pred = reg_model.predict(input_df)[0]

            st.success("Prediction complete")
            res1, res2 = st.columns(2)
            res1.metric("Predicted Eligibility", str(eligibility_pred))
            res2.metric("Max Monthly EMI", f"₹{max_emi_pred:,.2f}")

            add_record(name, age, monthly_salary, credit_score, requested_amount,
                       str(eligibility_pred), float(max_emi_pred))
            st.caption("This prediction has been saved to your records.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("This usually means the input needs the same preprocessing "
                    "(scaling/encoding) used during training. Load scaler.pkl and "
                    "label_encoders.pkl and apply them to input_df before predicting.")

# ============================================================
# PAGE: DATA EXPLORER
# ============================================================
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown("Explore your saved prediction records.")

    df = get_records()
    if df.empty:
        st.info("No records yet. Make a prediction first.")
    else:
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Eligibility Distribution")
            st.bar_chart(df["predicted_eligibility"].value_counts())
        with col2:
            st.subheader("Predicted Max EMI Distribution")
            st.line_chart(df["predicted_max_emi"])

# ============================================================
# PAGE: CRUD
# ============================================================
elif page == "Manage Records (CRUD)":
    st.title("Manage Applicant Records")

    df = get_records()

    st.subheader("All Records")
    if df.empty:
        st.info("No records yet.")
    else:
        st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("Update a Record")
    if not df.empty:
        record_id = st.selectbox("Select Record ID to Edit", df["id"].tolist())
        selected = df[df["id"] == record_id].iloc[0]

        with st.form("update_form"):
            u_name = st.text_input("Name", value=selected["name"])
            u_age = st.number_input("Age", value=int(selected["age"]))
            u_salary = st.number_input("Monthly Salary", value=float(selected["monthly_salary"]))
            u_credit = st.number_input("Credit Score", value=float(selected["credit_score"]))
            u_amount = st.number_input("Requested Amount", value=float(selected["requested_amount"]))
            submitted = st.form_submit_button("Update Record")
            if submitted:
                update_record(record_id, u_name, u_age, u_salary, u_credit, u_amount)
                st.success("Record updated.")
                st.rerun()

    st.divider()
    st.subheader("Delete a Record")
    if not df.empty:
        del_id = st.selectbox("Select Record ID to Delete", df["id"].tolist(), key="del_select")
        if st.button("Delete Record", type="secondary"):
            delete_record(del_id)
            st.success("Record deleted.")
            st.rerun()

# ============================================================
# PAGE: MODEL DASHBOARD
# ============================================================
elif page == "Model Dashboard":
    st.title("Model Performance Dashboard")
    st.markdown("Live metrics pulled from the MLflow Model Registry on DagsHub.")

    try:
        client = mlflow.tracking.MlflowClient()

        st.subheader("Classification Model — EMI_Eligibility_Classifier")
        clf_versions = client.search_model_versions("name='EMI_Eligibility_Classifier'")
        if clf_versions:
            latest_clf = max(clf_versions, key=lambda v: int(v.version))
            run = client.get_run(latest_clf.run_id)
            metrics = run.data.metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            c2.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
            c3.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")

        st.subheader("Regression Model — Max_EMI_Regressor")
        reg_versions = client.search_model_versions("name='Max_EMI_Regressor'")
        if reg_versions:
            latest_reg = max(reg_versions, key=lambda v: int(v.version))
            run = client.get_run(latest_reg.run_id)
            metrics = run.data.metrics
            c1, c2 = st.columns(2)
            c1.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
            c2.metric("R² Score", f"{metrics.get('r2', 0):.4f}")

        st.divider()
        st.markdown(f"[View full MLflow UI on DagsHub](https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments)")

    except Exception as e:
        st.error(f"Could not load dashboard metrics: {e}")
