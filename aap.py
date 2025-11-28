import joblib
import streamlit as st
import numpy as np

st.set_page_config(page_title="Nestle Stock Price Prediction", page_icon="📈")

# Title and header
st.title("📈 Nestle Stock Price Predictor")
st.header("Predict the closing stock price of Nestle using machine learning")

# Load the model
filename = "linear.joblib"
try:
    loaded_model = joblib.load(filename)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found! Please ensure 'linear.joblib' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# # Display model information
# with st.expander("📘 Model Information"):
#     st.write(f"**Model Type:** Multiple Linear Regression")
#     st.write(f"**Intercept:** {loaded_model.intercept_:.4f}")
#     st.write("**Coefficients:**")
#     for idx, coef in enumerate(loaded_model.coef_):
#         st.write(f"Feature {idx+1}: {coef:.4f}")

st.markdown("---")
st.subheader("📊 Enter the Feature Values")

open_price = st.number_input("Enter Open Price", min_value=0.0, step=0.1)
high_price = st.number_input("Enter High Price", min_value=0.0, step=0.1)
low_price = st.number_input("Enter Low Price", min_value=0.0, step=0.1)
no_of_shares = st.number_input("Enter Number of Shares", min_value=0.0, step=1.0)
no_of_trades = st.number_input("Enter Number of Trades", min_value=0.0, step=1.0)
deliverable_qty = st.number_input("Enter Deliverable Quantity", min_value=0.0, step=1.0)

# Feature array for prediction
features = np.array([[
    open_price,
    high_price,
    low_price,
    no_of_shares,
    no_of_trades,
    deliverable_qty
]])

# Predict button
btn = st.button("🔮 Predict Stock Price", type="primary", use_container_width=True)

if btn:
    if open_price == 0 and high_price == 0:
        st.warning("⚠️ Please enter valid feature values before prediction.")
    else:
        predicted_price = loaded_model.predict(features)[0]

        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Close Price", f"₹{predicted_price:.2f}")

        with col2:
            st.info("Prediction successful based on input stock indicators.")

        st.success(f"📌 Expected Nestle Closing Price: **₹{predicted_price:.2f}**")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit | Nestle Stock Price Prediction</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This application predicts **Nestle India's closing stock price**
    using a Multiple Linear Regression model.
    """)

    st.header("📋 Instructions")
    st.write("""
    1. Enter required stock features  
    2. Click on **Predict Stock Price**  
    3. View the predicted closing value  
    """)
