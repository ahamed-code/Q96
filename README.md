💹 Q96 – Investment & Dream Planner App
Q96 is a smart, user-friendly personal finance dashboard built with Streamlit and enhanced with AI-powered insights. It allows you to track investments, plan your dreams, and visualize your financial future — all in one place.

🚀 Features
📊 Investment Tracker
Upload your own CSV files to analyze your investment portfolio.

View allocation by asset class, performance over time, and trend analysis.

Visual guides and example images provided for beginners.

🎯 Dream Planning Toolkit
Turn your goals into reality:

🏠 Buy a House

🚗 Buy a Car

🎓 Child’s Education

Each goal includes:

Custom financial calculators

Adjustable expected return sliders

Smart monthly savings targets

Visual feedback with AI-generated tips

🤖 AI-Powered Assistance
Smart recommendations based on user input

Auto-filled projections for typical financial goals

Personalized financial suggestions using AI prompts

🔮 Financial Calculators
🎓 Education Planner – Estimate savings for your child’s future.

🏡 Home Planner – Plan for down payments and EMIs.

🚙 Car Planner – Get saving targets based on your preferences.

📈 Optional Market Integrations (Pluggable)
✅ Live Crypto Prices (via CoinMarketCap API) (optional)

✅ Stock Prices (via Yahoo Finance / RapidAPI) (optional)

🛠️ Tech Stack
Tool/Library	Purpose
Streamlit	UI + App Framework
Python	Core backend logic
Pandas	CSV and data manipulation
Matplotlib / Plotly	Financial chart visualizations
OpenAI / Gemini	AI-Powered insights (optional)
CoinMarketCap API	Crypto prices (optional)
Yahoo Finance API	Stock prices (optional)

📂 Example CSV Upload Format
A sample investment upload might look like this:

Date	Asset	Amount Invested	Current Value
2023-01-01	Mutual Fund A	50,000	57,000
2023-03-12	Bitcoin	10,000	12,500

📌 Like the above example, structure your CSV similarly for best results.

🖼️ Refer to the in-app image for guidance:

python
Copy
Edit
st.image('example.png', caption="Like the above example", use_column_width=True)
🚧 Future Plans
 User authentication & login

 Goal reminder notifications

 Dashboard saving/export options

 AI-driven full financial health report

