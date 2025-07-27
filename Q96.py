import streamlit as st
st.set_page_config(page_title="Q96 - Finance Planner", layout="wide")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import hashlib
import requests
import google.generativeai as genai

# --- Gemini AI Setup ---
genai.configure(api_key="AIzaSyCl2dDMsNeChqjZ_HkRW6ZCTIujtCzx2aI")
# --- Main Tab Layout ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Home", "SIP Recommendation", "Dream Goals", "Investment Tracker", "Retirement Planner", "Financial Score", "AI Financial Planner"
])

# --- Home ---
with tab1:
    st.header("Welcome to Q96 Investment Tracker")
    st.image('pexels-artempodrez-5716032.jpg', use_column_width=True)
    st.write("This app helps you track investments, recommend SIPs, and predict growth using AI.")

# --- SIP Recommendation ---
with tab2:
    st.header("SIP Recommendation")
    st.image('SIP.jpg', use_column_width=True)
    target = st.number_input("Target Amount (₹)", min_value=1000, step=500)
    sip = st.number_input("Monthly SIP (₹)", min_value=500, step=500)
    annual_return = st.slider("Expected Return (%)", 5, 20, 12)
    inflation = st.slider("Expected Inflation (%)", 1, 10, 5)
    years = st.slider("Duration (Years)", 1, 30, 10)

    real_rate = ((1 + annual_return/100) / (1 + inflation/100)) - 1
    m_rate = real_rate / 12

    if target and sip:
        future_value = sip * (((1 + m_rate)**(years*12) - 1) / m_rate) * (1 + m_rate)
        st.write(f"**Estimated Future Value:** ₹{future_value:,.2f}")
        df_growth = pd.DataFrame(
            [sip * (((1 + m_rate)**m - 1) / m_rate) * (1 + m_rate) for m in range(1, years*12 + 1)],
            index=range(1, years*12 + 1),
            columns=["Investment Value"]
        )
        st.line_chart(df_growth)

# --- Dream Goals ---
with tab3:
    st.header("Achieve Your Dreams")
    st.image('DREAMS.jpg', use_column_width=True)
    goal = st.selectbox("Select Goal", ["Child's Education", "Buy a House"])

    if goal == "Child's Education":
        st.subheader("🎓 Plan for Child's Education")
        years_until_college = st.slider("Years Until College", 5, 25, 10)
        future_cost = st.number_input("Estimated Future Cost (₹)", min_value=1000000)
        rate = st.slider("Expected Annual Return (%)", 6, 15, 10)
        monthly_save = future_cost / (((1 + rate/100/12)**(years_until_college*12) - 1) / (rate/100/12))
        st.success(f"Save ₹{monthly_save:,.0f}/month to achieve ₹{future_cost:,} in {years_until_college} years.")

    elif goal == "Buy a House":
        st.subheader("🏡 Plan to Buy a House")
        cost = st.number_input("Estimated House Cost (₹)", min_value=500000)
        down_payment = st.slider("Down Payment (%)", 10, 100, 20)
        duration = st.slider("Saving Period (Years)", 1, 20, 5)
        needed = cost * down_payment / 100
        monthly_save = needed / (duration * 12)
        st.info(f"To make a ₹{needed:,.0f} down payment in {duration} years, save ₹{monthly_save:,.0f}/month.")


# --- Investment Tracker ---
with tab4:
    st.header("Track Your Investments")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.image('example.png', use_column_width=True)
    st.markdown("<h5 style='text-align: center;'>📄 Like the above example</h5>", unsafe_allow_html=True)
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success("File uploaded!")
            st.dataframe(df.head())

            if "Date" in df:
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                df.dropna(subset=["Date"], inplace=True)
                df.sort_values("Date", inplace=True)
                st.line_chart(df.set_index("Date"))

            if "Amount" in df:
                total = df["Amount"].sum()
                st.write(f"**Total Invested:** ₹{total:,.2f}")
                if st.button("Predict Future Growth"):
                    df_model = df.copy()
                    df_model['Year'] = df_model['Date'].dt.year
                    df_group = df_model.groupby("Year")["Amount"].sum().reset_index()
                    X = df_group[['Year']]
                    y = df_group['Amount']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = xgb.XGBRegressor()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    rmse = mean_squared_error(y_test, preds, squared=False)
                    st.write(f"Model RMSE: ₹{rmse:,.2f}")
                    future_years = pd.DataFrame({'Year': range(df_group['Year'].max()+1, df_group['Year'].max()+11)})
                    future_preds = model.predict(future_years)
                    st.line_chart(pd.DataFrame(future_preds, index=future_years['Year'], columns=["Predicted"]))
        except Exception as e:
            st.error(f"Error: {e}")

# --- Retirement Planner ---
with tab5:
    st.header("Retirement Planner")
    current_age = st.slider("Current Age", 18, 60, 30)
    retirement_age = st.slider("Retirement Age", current_age+1, 70, 60)
    monthly_expenses = st.number_input("Monthly Expenses Now (₹)", min_value=1000)
    inflation = st.slider("Inflation Rate (%)", 1, 10, 5)
    post_retire_years = st.slider("Years after Retirement", 10, 40, 25)
    return_rate = st.slider("Expected Return Rate (%)", 6, 15, 10)
    future_expense = monthly_expenses * ((1 + inflation / 100) ** (retirement_age - current_age))
    needed = future_expense * 12 * post_retire_years
    monthly_saving = needed / (((1 + return_rate / 100 / 12)**((retirement_age - current_age)*12) - 1) / (return_rate / 100 / 12))
    st.info(f"To retire at {retirement_age} with expenses of ₹{future_expense:,.0f}/month, save ₹{monthly_saving:,.0f}/month.")

# --- Financial Score ---
with tab6:
    st.header("Your Financial Health Score 🧠")
    income = st.number_input("Monthly Income (₹)", min_value=0)
    expenses = st.number_input("Monthly Expenses (₹)", min_value=0)
    emi = st.number_input("Monthly EMI (₹)", min_value=0)
    investments = st.number_input("Total Investments (₹)", min_value=0)
    savings_rate = (income - expenses - emi) / income if income > 0 else 0
    score = 0
    if savings_rate > 0.3: score += 40
    elif savings_rate > 0.2: score += 30
    elif savings_rate > 0.1: score += 20
    if investments > income * 12: score += 40
    elif investments > income * 6: score += 30
    elif investments > income * 3: score += 20
    if emi < income * 0.3: score += 20
    st.success(f"🧠 Financial Health Score: {score}/100")
    if score >= 80:
        st.balloons()
        st.markdown("✅ Excellent! Keep it up!")
    elif score >= 60:
        st.markdown("🟡 Good, but room to grow.")
    else:
        st.markdown("🔴 Needs attention. Consider saving more.")

 
# --- AI Financial Planner (Together AI) ---
with tab7:
    st.header("🤖 AI-Powered Financial Planning ")
    user_input = st.text_area("Describe your financial situation and goals:", placeholder="I am 28 years old, earning ₹80,000/month, want to save for a house and a car.")
    email = st.text_input("Optional Email (to receive summary)")
    dreams = st.text_area("Dreams (Optional):", placeholder="e.g. Own a BMW, fund child's college, retire at 50")

    if st.button("Generate AI Plan"):
        if user_input:
            with st.spinner("Generating plan..."):
                try:
                    TOGETHER_API_KEY = "e5c460a5079190b146122f597c686c3f8d0240a52bc744f9ec3e02ada12c4c98"
                    url = "https://api.together.xyz/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {TOGETHER_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    prompt = f"""You're a certified Indian financial advisor. A user says:
Situation: {user_input}
Dreams: {dreams if dreams else "Not provided"}
Give a clear, personalized plan including:
- Budgeting tips
- Investment options (SIPs, FD, MF)
- Retirement strategy
- Timeline for dreams
- Risk management advice"""

                    payload = {
                        "model": "meta-llama/Llama-3-70b-chat-hf",
                        "messages": [
                            {"role": "system", "content": "You are a helpful financial advisor."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1024
                    }

                    res = requests.post(url, headers=headers, json=payload)
                    res.raise_for_status()
                    result = res.json()
                    ai_text = result["choices"][0]["message"]["content"]

                    st.success("✅ Plan Generated:")
                    st.markdown(ai_text)
                    if email:
                        st.info(f"Plan will be sent to {email} (email feature coming soon!)")
                except Exception as e:
                    st.error(f"❌ AI API Error: {e}")
        else:
            st.warning("Please describe your financial situation first.")
