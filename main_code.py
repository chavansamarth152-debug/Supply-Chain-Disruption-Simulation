import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Supply Chain DSS", layout="wide")

st.title("📦 AI-Powered Supply Chain Disruption DSS")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Using default sample data")
    df = pd.read_csv("sample_data.csv")

st.subheader("📊 Raw Data")
st.write(df)

# -------------------------------
# PREPROCESSING
# -------------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# Feature Engineering
df["Risk_Score"] = (1 - df["Reliability"]) * df["Lead_Time"]

# -------------------------------
# SCENARIO SELECTION
# -------------------------------
st.sidebar.header("⚙️ Simulation Controls")

scenario = st.sidebar.selectbox("Select Scenario", [
    "None",
    "Flood (Supplier Delay)",
    "Demand Spike",
    "Transport Strike (Cost Increase)"
])

demand_increase = st.sidebar.slider("Demand Increase %", 0, 100, 20)
delay_days = st.sidebar.slider("Delay Days", 0, 10, 2)

# -------------------------------
# SIMULATION ENGINE
# -------------------------------
sim_df = df.copy()

if scenario == "Flood (Supplier Delay)":
    sim_df["Lead_Time"] += delay_days
    sim_df["Disruption"] = "Flood"

elif scenario == "Demand Spike":
    sim_df["Demand"] *= (1 + demand_increase / 100)
    sim_df["Disruption"] = "Demand Spike"

elif scenario == "Transport Strike (Cost Increase)":
    sim_df["Transport_Cost"] *= 1.3
    sim_df["Disruption"] = "Strike"

# -------------------------------
# MACHINE LEARNING MODELS
# -------------------------------

# Demand Prediction
X = df[["Inventory", "Safety_Stock", "Lead_Time", "Transport_Cost"]]
y = df["Demand"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)

pred_demand = lr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_demand))

# Delay Classification
df["Delay_Label"] = (df["Lead_Time"] > 6).astype(int)

X2 = df[["Reliability", "Transport_Cost"]]
y2 = df["Delay_Label"]

dt = DecisionTreeClassifier()
dt.fit(X2, y2)

# Risk Classification
df["Risk_Label"] = pd.cut(df["Risk_Score"],
                         bins=[-1, 2, 5, 10],
                         labels=[0, 1, 2])

rf = RandomForestClassifier()
rf.fit(X2, df["Risk_Label"])

# -------------------------------
# DECISION ENGINE
# -------------------------------
recommendations = []

for i, row in sim_df.iterrows():

    if row["Inventory"] < row["Demand"]:
        recommendations.append("Increase Inventory")

    if row["Reliability"] < 0.7:
        recommendations.append("Switch Supplier")

    if row["Transport_Cost"] > 65:
        recommendations.append("Optimize Transport Route")

# Remove duplicates
recommendations = list(set(recommendations))

# -------------------------------
# COST ANALYSIS
# -------------------------------
original_cost = (df["Transport_Cost"] + df["Product_Cost"]).sum()
new_cost = (sim_df["Transport_Cost"] + sim_df["Product_Cost"]).sum()

# -------------------------------
# OUTPUT SECTION
# -------------------------------
st.header("📈 Results")

col1, col2 = st.columns(2)

with col1:
    st.metric("RMSE (Demand Prediction)", round(rmse, 2))
    st.metric("Original Cost", round(original_cost, 2))

with col2:
    st.metric("New Cost", round(new_cost, 2))
    st.metric("Cost Difference", round(new_cost - original_cost, 2))

# -------------------------------
# GRAPHS
# -------------------------------

st.subheader("📊 Demand vs Inventory")

fig1 = px.bar(sim_df, x="Supplier_ID", y=["Demand", "Inventory"],
              barmode="group")
st.plotly_chart(fig1)

st.subheader("💰 Cost Comparison")

fig2 = px.bar(x=["Original", "New"], y=[original_cost, new_cost])
st.plotly_chart(fig2)

st.subheader("📦 Inventory Trend")

fig3 = px.line(sim_df, x="Supplier_ID", y="Inventory")
st.plotly_chart(fig3)

st.subheader("🔥 Risk Heatmap")

fig4 = px.scatter(sim_df, x="Reliability", y="Lead_Time",
                  color="Risk_Score", size="Inventory")
st.plotly_chart(fig4)

# -------------------------------
# RECOMMENDATIONS
# -------------------------------
st.subheader("🧠 AI Recommendations")

for rec in recommendations:
    st.success(rec)

# -------------------------------
# REPORT
# -------------------------------
st.subheader("📄 Report Summary")

st.write(f"""
Scenario: {scenario}

- Demand Prediction RMSE: {round(rmse,2)}
- Original Cost: {original_cost}
- New Cost: {new_cost}

Recommendations:
{recommendations}
""")

# -------------------------------
# DOWNLOAD REPORT
# -------------------------------
report = sim_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Report CSV",
    data=report,
    file_name='report.csv',
    mime='text/csv'
)
