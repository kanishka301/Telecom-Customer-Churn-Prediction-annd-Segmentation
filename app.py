import pandas as pd
import numpy as np
import pickle
import os
import streamlit as st
import plotly.express as px

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Telecom Churn Dashboard",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Telecom Customer Churn Dashboard")
st.caption("Predict churn risk, explore the data, and view customer segments.")


# ============================================================
# LOAD MODEL BUNDLE (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model_bundle():
    path = os.path.join("model", "model_bundle.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# LOAD DATA (cached so it only loads once)
# ============================================================
@st.cache_data
def load_raw_data():
    path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(path)
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    return df


@st.cache_data
def load_segment_data():
    path = os.path.join("data", "Final_Segments.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


bundle = load_model_bundle()
raw_df = load_raw_data()
segment_df = load_segment_data()

# ============================================================
# TABS
# ============================================================
tab1,tab2,tab3=st.tabs(['🎯Predict Churn','📊Explore Data','👥Customer Segments'])

# ------------------------------------------------------------
# TAB 1: PREDICT CHURN
# ------------------------------------------------------------
with tab1:
    st.subheader("Predict Churn for a Single Customer")

    if bundle is None:
        st.error(
            "Model not found. Please run `python train_model.py` first "
            "to create model/model_bundle.pkl."
        )
    else:
        model = bundle["model"]
        scaler = bundle["scaler"]
        encoders = bundle["encoders"]
        feature_names = bundle["feature_names"]

        st.write("Fill in the customer's details below:")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )

        with col2:
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )

        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0
            )
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, value=840.0, step=1.0
            )

        st.write("")

        if st.button("Predict Churn", type="primary"):
            # Build a single-row dataframe matching training column order
            input_data = {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            input_df = pd.DataFrame([input_data])[feature_names]

            # Encode categorical columns using the SAME encoders used in training
            for col, encoder in encoders.items():
                input_df[col] = encoder.transform(input_df[col])

            # Scale using the SAME scaler used in training (fixes the original bug!)
            input_scaled = scaler.transform(input_df)

            probability = model.predict_proba(input_scaled)[0][1]
            BEST_THRESHOLD = 0.63 #from model_training
            prediction = int(probability >= BEST_THRESHOLD)

            st.write("---")
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                if prediction ==1:
                    st.error("⚠️ This customer is LIKELY TO CHURN")
                else:
                    st.success("This customer is LIKELY TO STAY")

            with result_col2:
                st.metric("Churn Probability", f"{probability*100:.2f}%")

            st.progress(float(probability))

# TAB 2: EXPLORE DATA (EDA)
with tab2:
    st.subheader("Explore the Dataset")

    total_customers = len(raw_df)
    churned = (raw_df["Churn"] == "Yes").sum()
    churn_rate = churned / total_customers

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers", f"{total_customers:,}")
    m2.metric("Churned Customers", f"{churned:,}")
    m3.metric("Overall Churn Rate", f"{churn_rate:.1%}")

    st.write("---")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.pie(
            raw_df, names="Churn", title="Churn Distribution", hole=0.4,
            color="Churn", color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            raw_df, x="tenure", color="Churn", barmode="overlay",
            title="Tenure Distribution by Churn", nbins=30,
            color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.histogram(
            raw_df, x="Contract", color="Churn", barmode="group",
            title="Churn by Contract Type",
            color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View raw data"):
        st.dataframe(raw_df, use_container_width=True)

with tab3:
    st.subheader("Customer Segments")

    if segment_df is None:
        st.warning(
            "data/Final_Segments.csv not found. Add it to the data folder "
            "to see customer segments here."
        )
    else:
        seg_counts = segment_df["Segment_Name"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        c1, c2 = st.columns([1, 1])

        with c1:
            fig = px.bar(
                seg_counts, x="Segment", y="Count", color="Segment",
                title="Number of Customers per Segment",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.pie(
                segment_df, names="Segment_Name", title="Segment Share",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        st.write("**Average profile of each segment:**")
        profile = segment_df.groupby("Segment_Name")[
            ["tenure", "MonthlyCharges", "TotalCharges", "Churn_Probability"]
        ].mean().round(2)
        st.dataframe(profile, use_container_width=True)

        st.write("---")