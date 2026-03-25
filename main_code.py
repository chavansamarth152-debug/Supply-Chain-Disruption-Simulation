# =========================
# AI SUPPLY CHAIN DSS APP
# =========================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🚀 AI-Powered Supply Chain DSS")

# =========================
# DATA GENERATION
# =========================
@st.cache_data
def generate_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "supplier_id": np.arange(1, 101),
        "location": np.random.choice(["India", "China", "USA"], 100),
        "reliability": np.random.uniform(0.7, 1.0, 100),
        "inventory": np.random.randint(100, 500, 100),
        "demand": np.random.randint(80, 400, 100),
        "transport_cost": np.random.randint(100, 500, 100),
        "lead_time": np.random.randint(1, 10, 100),
        "product_cost": np.random.randint(50, 200, 100)
    })
    return data

df = generate_data()

# =========================
# PREPROCESSING
# =========================
def preprocess(df):
    df = df.copy()
    df["risk_score"] = (1 - df["reliability"]) * df["lead_time"]
    scaler = StandardScaler()
    df[["inventory","demand","transport_cost"]] = scaler.fit_transform(
        df[["inventory","demand","transport_cost"]]
    )
    return df

df_processed = preprocess(df)

# =========================
# ML MODELS
# =========================
def train_models(df):
    X = df[["inventory","transport_cost","lead_time","reliability"]]
    y_demand = df["demand"]
    y_delay = df["lead_time"]
    y_risk = (df["risk_score"] > df["risk_score"].median()).astype(int)

    demand_model = LinearRegression().fit(X, y_demand)
    delay_model = DecisionTreeRegressor().fit(X, y_delay)
    risk_model = RandomForestClassifier().fit(X, y_risk)

    return demand_model, delay_model, risk_model

demand_model, delay_model, risk_model = train_models(df_processed)

# =========================
# UI INPUTS
# =========================
st.sidebar.header("⚙️ Scenario Controls")

scenario = st.sidebar.selectbox("Select Scenario", [
    "Normal",
    "Flood (Supplier Delay)",
    "Demand Spike",
    "Transport Strike"
])

demand_increase = st.sidebar.slider("Demand Increase %", 0, 100, 20)
delay_days = st.sidebar.slider("Delay Days", 0, 10, 3)
cost_increase = st.sidebar.slider("Cost Increase %", 0, 100, 25)

# =========================
# SIMULATION ENGINE
# =========================
def simulate(df, scenario):
    df = df.copy()

    if scenario == "Flood (Supplier Delay)":
        df["lead_time"] += delay_days

    elif scenario == "Demand Spike":
        df["demand"] *= (1 + demand_increase/100)

    elif scenario == "Transport Strike":
        df["transport_cost"] *= (1 + cost_increase/100)

    return df

df_sim = simulate(df_processed, scenario)

# =========================
# PREDICTIONS
# =========================
X_sim = df_sim[["inventory","transport_cost","lead_time","reliability"]]

pred_demand = demand_model.predict(X_sim)
pred_delay = delay_model.predict(X_sim)
pred_risk = risk_model.predict(X_sim)

df_sim["predicted_demand"] = pred_demand
df_sim["predicted_delay"] = pred_delay
df_sim["risk"] = pred_risk

# =========================
# DECISION ENGINE
# =========================
def decisions(df):
    rec = []

    if df["risk"].mean() > 0.5:
        rec.append("🔴 High Risk: Switch supplier")

    if df["predicted_delay"].mean() > 5:
        rec.append("⏱ Use faster transport")

    if df["predicted_demand"].mean() > df["inventory"].mean():
        rec.append("📦 Increase inventory")

    if not rec:
        rec.append("✅ System Stable")

    return rec

recommendations = decisions(df_sim)

# =========================
# VISUALIZATION
# =========================
st.header("📊 Dashboard")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(df_sim, x="inventory", y="predicted_demand",
                      title="Demand vs Inventory")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(df_sim, x="predicted_delay",
                        title="Delay Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# Risk Heatmap
st.subheader("🔥 Risk Heatmap")
fig3 = px.density_heatmap(df_sim, x="inventory", y="predicted_demand")
st.plotly_chart(fig3, use_container_width=True)

# =========================
# COST ANALYSIS
# =========================
st.header("💰 Cost Analysis")

base_cost = df["transport_cost"].sum()
new_cost = df_sim["transport_cost"].sum()

st.metric("Original Cost", f"${base_cost:.2f}")
st.metric("After Simulation", f"${new_cost:.2f}")
st.metric("Cost Change", f"{(new_cost-base_cost):.2f}")

# =========================
# OUTPUT
# =========================
st.header("🧠 AI Decision Support")

for r in recommendations:
    st.success(r)

risk_level = "Low"
if df_sim["risk"].mean() > 0.6:
    risk_level = "High"
elif df_sim["risk"].mean() > 0.3:
    risk_level = "Medium"

st.subheader(f"⚠️ Risk Level: {risk_level}")

# =========================
# DOWNLOAD REPORT
# =========================
st.subheader("⬇️ Download Results")

csv = df_sim.to_csv(index=False)
st.download_button("Download CSV", csv, "results.csv")
