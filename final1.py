
import joblib
import pandas as pd
import streamlit as st


from tensorflow import keras
import keras

model = keras.models.load_model("churn_model.keras", compile=False)


scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # The exact feature order from training

# Streamlit UI
st.title("📊 Customer order status Prediction")
st.write("Enter order details to predict churn (0 = Success Order is Delivery , 1 = Order is Pending Coming Soon).")

# User inputs
book_id = st.number_input("Book ID", min_value=0, step=1)
order_dayofweek = st.number_input("Order Day of Week", min_value=0, max_value=6, step=1)
order_month = st.number_input("Order Month", min_value=1, max_value=12, step=1)
order_year = st.number_input("Order Year", min_value=2000, max_value=2100, step=1)
total_orders = st.number_input("Total Orders", min_value=0, step=1)
price = st.number_input("Price", min_value=0.0, step=0.01)

# Predict button
if st.button("Predict Churn"):
    # Prepare input DataFrame
    input_data = pd.DataFrame([{
        "book_id": book_id,
        "order_dayofweek": order_dayofweek,
        "order_month": order_month,
        "order_year": order_year,
        "total_orders": total_orders,
        "price": price
    }])

    # Ensure all features used in training exist in the same order
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Fill missing with 0 or suitable default

    input_data = input_data[feature_columns]

    # Scale numeric features
    input_data = pd.DataFrame(scaler.transform(input_data), columns=feature_columns)

    # Predict churn probability
    probability = model.predict(input_data)[0][0]
    prediction = 1 if probability >= 0.5 else 0

    # Output results
    st.write(f"**Prediction:** {'Order is Pending Coming Soon' if prediction == 1 else 'Success Order is Delivery'}")
    st.write(f"**Churn Probability:** {probability:.2f}")
