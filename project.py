import streamlit as st
import pickle
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Attrition Predictor", page_icon="🧑‍💼", layout="centered")

# --- Load Model ---
with open("Employee_Attrition_Model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/color/96/employee.png", width=80)
st.sidebar.title("Employee Attrition Project")
st.sidebar.markdown(
    """
    **Predict if an employee will leave or stay!**  
    - Model: Logistic Regression  
    - Features:  
        - Age  
        - Years at Company  
        - Job Satisfaction  
        - Distance From Home  
        - Monthly Income  
    """
)
st.sidebar.info("MANNALA ABIRAM")

# --- Main Title ---
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🔮 Employee Attrition Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)
st.write("Enter employee details below:")

# --- Input Form ---
with st.form("attrition_form"):
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("🎂 Age", min_value=18, max_value=70, value=30)
        YearsAtCompany = st.number_input("🏢 Years at Company", min_value=0, max_value=50, value=5)
        MonthlyIncome = st.number_input("💰 Monthly Income", min_value=1000, max_value=200000, value=5000, step=500)
    with col2:
        JobSatisfaction = st.selectbox("😊 Job Satisfaction", options=[0,1,2,3])
        DistanceFromHome = st.number_input("🚗 Distance From Home (km)", min_value=0, max_value=100, value=10)

    submitted = st.form_submit_button("🔎 Predict Attrition")

# --- Prediction ---
if submitted:
    input_df = pd.DataFrame({
        "Age": [Age],
        "YearsAtCompany": [YearsAtCompany],
        "MonthlyIncome": [MonthlyIncome],
        "JobSatisfaction": [JobSatisfaction],
        "DistanceFromHome": [DistanceFromHome]  
    })
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ Low Risk: Employee likely to *stay*. (Probability of staying: {prob[1]:.2f})")
        st.balloons()
    else:
        st.error(f"⚠ High Risk: Employee likely to *leave*. (Probability of leaving: {prob[0]:.2f})")
        st.snow()

    # --- Probability Chart ---
    st.markdown("### 📊 Probability Breakdown")
    st.progress(prob[1] if prediction == 1 else prob[0])
    st.markdown(
        f"""
        - **Probability of Staying:** `{prob[1]:.2f}`
        - **Probability of Leaving:** `{prob[0]:.2f}`
        """
    )
    