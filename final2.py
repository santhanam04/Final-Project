import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = tf.keras.models.load_model("churn_ann_model.h5")
scaler = joblib.load("scaler.joblib")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("transactions.csv")
df1 = pd.read_csv("customer_metrics.csv")

# Merge on customer_id
merged_df = pd.merge(df1, df, on="customer_id", how="inner")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Customer Churn Prediction")

# --- Input values ---
total_orders = st.number_input("Total Orders", min_value=0, value=0, step=1)
total_spend = st.number_input("Total Spend ($)", min_value=0.0, value=0.0, step=10.0)
avg_order_value = st.number_input("Average Order Value ($)", min_value=0.0, value=0.0, step=1.0,format="%.6f")
recency_days = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)

if st.button("Predict Churn"):
    # -----------------------------
    # Prepare input for model
    # -----------------------------
    input_data = pd.DataFrame([{
        "total_orders": total_orders,
        "total_spend": total_spend,
        "avg_order_value": avg_order_value,
        "recency_days": recency_days
    }])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_scaled)[0][0]
    churn_pred = 1 if prediction >= 0.5 else 0

    # -----------------------------
    # Fetch same customer row (closest match)
    # -----------------------------
    customer_match = merged_df[
        (merged_df["total_orders"] == total_orders) &
        (merged_df["total_spend"] == total_spend) &
        (merged_df["avg_order_value"] == avg_order_value) &
        (merged_df["recency_days"] == recency_days)
    ]

    if not customer_match.empty:
        customer_details = customer_match[[
            "customer_id", "first_name", "last_name", "email",
            "total_orders", "total_spend", "avg_order_value", "recency_days"
        ]].drop_duplicates()

        # Add churn prediction
        customer_details["Predicted_Churn"] = "Yes" if churn_pred == 1 else "No"

        # Display table
        st.subheader("Customer Details with Prediction")
        st.dataframe(customer_details)
    else:
        st.warning("No exact customer found with these values. Showing only prediction result.")
        st.write("Predicted Churn:", "Yes" if churn_pred == 1 else "No")
   