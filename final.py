import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
# -----------------------------
# Load Data, Model, and Scaler
# -----------------------------
# Load customer metrics (preprocessed dataset with churn column)
customer_metrics = pd.read_csv("customer_metrics.csv")

# Load trained ANN model
model = tf.keras.models.load_model("churn_ann_model.h5")

# Load scaler

scaler = joblib.load("scaler.joblib")

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_books(customer_id, n=3):
    """Simple recommendation: top popular books not yet bought by the customer"""
    # Load original transaction data
    df = pd.read_csv("transactions.csv")  
    
    # Books already purchased by this customer
    purchased = df[df['customer_id'] == customer_id]['book_id'].unique()
    
    # Top-selling books overall
    top_books = (
        df.groupby(['book_id', 'title'])
        .size()
        .reset_index(name='purchase_count')
        .sort_values('purchase_count', ascending=False)
    )
    
    # Exclude already purchased books
    recommendations = top_books[~top_books['book_id'].isin(purchased)].head(n)
    
    return recommendations[['title', 'purchase_count']]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Customer Churn & Recommendations", layout="centered")

st.title("📚 Customer Churn Prediction & Book Recommendations")

# Input Customer ID
customer_id = st.number_input("Enter Customer ID:", min_value=1, step=1)

if st.button("Get Prediction & Recommendations"):
    if customer_id in customer_metrics['customer_id'].values:
        # Get customer features
        customer_row = customer_metrics[customer_metrics['customer_id'] == customer_id]
        features = ['total_orders', 'total_spend', 'avg_order_value', 'recency_days']
        X = customer_row[features].values
        X_scaled = scaler.transform(X)

        # Predict churn
        churn_prob = model.predict(X_scaled)[0][0]
        churn_pred = 1 if churn_prob > 0.5 else 0

        # Display churn prediction
        st.subheader("🔮 Churn Prediction")
        st.write(f"**Churn Probability:** {churn_prob:.2f}")
        if churn_pred == 1:
            st.error("⚠️ This customer is likely to churn!")
        else:
            st.success("✅ This customer is likely to stay.")

        # Show recommendations
        st.subheader("📖 Recommended Books")
        recs = recommend_books(customer_id, n=3)
        st.table(recs)
    else:
        st.warning("Customer ID not found in dataset.")
