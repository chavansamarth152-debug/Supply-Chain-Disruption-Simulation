import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

import plotly.express as px

st.set_page_config(page_title="Supply Chain DSS", layout="wide")
st.title("📦 AI-Powered Supply Chain DSS (Error-Free Version)")

# -------------------------------
# LOAD DATA
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Using fallback sample data")
    df = pd.DataFrame({
        "Supplier_ID": ["S1","S2","S3"],
        "Reliability": [0.9,0.7,0.8],
        "Inventory": [500,300,400],
        "Demand": [450,400,420],
        "Transport_Cost": [50,60,55],
        "Lead_Time": [5,7,6],
        "Product_Cost": [20,18,22]
    })

st.write("### 📊 Data Preview")
st.write(df)

# -------------------------------
# AUTO COLUMN HANDLING (KEY FIX)
# -------------------------------
def get_col(name, default):
    return df[name] if name in df.columns else pd.Series([default]*len(df))

reliability = get_col("Reliability", 0.8)
inventory = get_col("Inventory", 300)
demand = get_col("Demand", 300)
transport = get_col("Transport_Cost", 50)
lead_time = get_col("Lead_Time", 5)
product_cost = get_col("Product_Cost", 20)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["Reliability"] = reliability
df["Inventory"] = inventory
df["Demand"] = demand
df["Transport_Cost"] = transport
df["Lead_Time"] = lead_time
df["Product_Cost"] = product_cost

df["Risk_Score"] = (1 - df["Reliability"]) * df["Lead_Time"]

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("⚙️ Controls")

scenario = st.sidebar.selectbox("Scenario", [
    "None",
    "Flood",
    "Demand Spike",
    "Transport Strike"
])

demand_increase = st.sidebar.slider("Demand Increase %", 0, 100, 20)
delay_days = st.sidebar.slider("Delay Days", 0, 10, 2)

# -------------------------------
# SIMULATION ENGINE
# -------------------------------
sim_df = df.copy()

if scenario == "Flood":
    sim_df["Lead_Time"] += delay_days

elif scenario == "Demand Spike":
    sim_df["Demand"] *= (1 + demand_increase / 100)

elif scenario == "Transport Strike":
    sim_df["Transport_Cost"] *= 1.3

# -------------------------------
# ML MODEL (SAFE)
# -------------------------------
features = ["Inventory", "Lead_Time", "Transport_Cost"]

X = sim_df[features]
y = sim_df["Demand"]

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

except:
    rmse = 0

# -------------------------------
# DECISION ENGINE
# -------------------------------
recommendations = []

if (sim_df["Inventory"] < sim_df["Demand"]).any():
    recommendations.append("Increase Inventory")

if (sim_df["Reliability"] < 0.7).any():
    recommendations.append("Switch Supplier")

if (sim_df["Transport_Cost"] > 60).any():
    recommendations.append("Optimize Transport")

if len(recommendations) == 0:
    recommendations.append("System is Stable")

# -------------------------------
# COST ANALYSIS
# -------------------------------
original_cost = (df["Transport_Cost"] + df["Product_Cost"]).sum()
new_cost = (sim_df["Transport_Cost"] + sim_df["Product_Cost"]).sum()

# -------------------------------
# OUTPUT
# -------------------------------
st.header("📈 Results")

col1, col2 = st.columns(2)

col1.metric("RMSE", round(rmse, 2))
col1.metric("Original Cost", round(original_cost, 2))

col2.metric("New Cost", round(new_cost, 2))
col2.metric("Cost Change", round(new_cost - original_cost, 2))

# -------------------------------
# GRAPHS
# -------------------------------
st.subheader("Demand vs Inventory")
fig1 = px.bar(sim_df, y=["Demand", "Inventory"])
st.plotly_chart(fig1)

st.subheader("Cost Comparison")
fig2 = px.bar(x=["Original", "New"], y=[original_cost, new_cost])
st.plotly_chart(fig2)

st.subheader("Risk Visualization")
fig3 = px.scatter(sim_df, x="Reliability", y="Lead_Time",
                  color="Risk_Score", size="Inventory")
st.plotly_chart(fig3)

# -------------------------------
# RECOMMENDATIONS
# -------------------------------
st.subheader("🧠 Recommendations")
for r in recommendations:
    st.success(r)

# -------------------------------
# REPORT DOWNLOAD
# -------------------------------
csv = sim_df.to_csv(index=False).encode('utf-8')

st.download_button("Download Report", csv, "report.csv", "text/csv")
