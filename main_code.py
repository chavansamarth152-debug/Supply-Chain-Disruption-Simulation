import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

st.set_page_config(layout="wide")
st.title("🚀 AI Supply Chain Disruption DSS")

# =========================
# DATA GENERATION
# =========================
def generate_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "supplier_id": np.random.randint(1, 5, 200),
        "location": np.random.choice(["North", "South", "East", "West"], 200),
        "reliability": np.random.uniform(0.7, 1.0, 200),
        "inventory": np.random.randint(50, 300, 200),
        "demand": np.random.randint(40, 250, 200),
        "transport_cost": np.random.randint(10, 100, 200),
        "lead_time": np.random.randint(1, 10, 200)
    })

    data["delay"] = data["lead_time"] + np.random.randint(0, 5, 200)
    data["risk"] = np.where(data["delay"] > 8, 1, 0)
    return data

# =========================
# LOAD DATA
# =========================
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = generate_data()

st.subheader("📊 Data Preview")
st.dataframe(df.head())

# =========================
# PREPROCESSING
# =========================
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df[["inventory", "demand", "transport_cost", "lead_time"]]
y_delay = df["delay"]
y_risk = df["risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# ML MODELS
# =========================
delay_model = DecisionTreeRegressor()
delay_model.fit(X_scaled, y_delay)

risk_model = RandomForestClassifier()
risk_model.fit(X_scaled, y_risk)

demand_model = LinearRegression()
demand_model.fit(df[["inventory"]], df["demand"])

# =========================
# UI INPUTS
# =========================
st.sidebar.header("⚙️ Simulation Controls")

scenario = st.sidebar.selectbox("Select Scenario", [
    "Normal",
    "Flood (Supplier Delay)",
    "Demand Spike",
    "Transport Strike"
])

demand_increase = st.sidebar.slider("Demand Increase %", 0, 100, 20)
delay_increase = st.sidebar.slider("Delay Increase Days", 0, 10, 3)

# =========================
# SIMULATION ENGINE
# =========================
df_sim = df.copy()

if scenario == "Flood (Supplier Delay)":
    df_sim["lead_time"] += delay_increase

elif scenario == "Demand Spike":
    df_sim["demand"] += df_sim["demand"] * (demand_increase / 100)

elif scenario == "Transport Strike":
    df_sim["transport_cost"] *= 1.5

# =========================
# PREDICTIONS
# =========================
X_sim = scaler.transform(df_sim[["inventory", "demand", "transport_cost", "lead_time"]])

pred_delay = delay_model.predict(X_sim)
pred_risk = risk_model.predict(X_sim)

df_sim["pred_delay"] = pred_delay
df_sim["pred_risk"] = pred_risk

# =========================
# DECISION ENGINE
# =========================
recommendations = []

if df_sim["pred_risk"].mean() > 0.5:
    recommendations.append("⚠️ High Risk → Use alternate supplier")
else:
    recommendations.append("✅ Risk Low → Continue current plan")

if df_sim["demand"].mean() > df["inventory"].mean():
    recommendations.append("📦 Increase inventory levels")

if scenario == "Transport Strike":
    recommendations.append("🚚 Change transport route")

# =========================
# COST ANALYSIS
# =========================
base_cost = df["transport_cost"].sum()
new_cost = df_sim["transport_cost"].sum()

# =========================
# OUTPUT
# =========================
st.header("📈 Results")

col1, col2, col3 = st.columns(3)
col1.metric("Avg Delay", f"{df_sim['pred_delay'].mean():.2f}")
col2.metric("Risk Level", "High" if df_sim["pred_risk"].mean() > 0.5 else "Low")
col3.metric("Cost Change", f"{new_cost - base_cost:.2f}")

# =========================
# VISUALIZATION
# =========================
st.subheader("📊 Demand vs Inventory")
fig1 = px.scatter(df_sim, x="inventory", y="demand", color="pred_risk")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📊 Cost Comparison")
cost_df = pd.DataFrame({
    "Type": ["Before", "After"],
    "Cost": [base_cost, new_cost]
})
fig2 = px.bar(cost_df, x="Type", y="Cost", color="Type")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🔥 Risk Heatmap")
heat = df_sim.pivot_table(values="pred_risk", index="supplier_id", columns="location")
st.dataframe(heat)

# =========================
# RECOMMENDATIONS
# =========================
st.header("🧠 AI Recommendations")

for rec in recommendations:
    st.success(rec)

# =========================
# REPORT
# =========================
st.header("📄 Report Summary")

st.write(f"""
Scenario: {scenario}

Average Delay: {df_sim['pred_delay'].mean():.2f}

Risk Level: {"High" if df_sim["pred_risk"].mean() > 0.5 else "Low"}

Cost Change: {new_cost - base_cost:.2f}

Recommendations:
{chr(10).join(recommendations)}
""")
