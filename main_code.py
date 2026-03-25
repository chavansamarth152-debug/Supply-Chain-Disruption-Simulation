import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("🚀 AI Supply Chain Disruption DSS")

# =========================
# SAFE DATA LOADER
# =========================
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except:
        return None

# =========================
# AUTO DATA GENERATOR
# =========================
def generate_data():
    np.random.seed(42)
    df = pd.DataFrame({
        "supplier_id": np.random.randint(1, 5, 100),
        "inventory": np.random.randint(50, 300, 100),
        "demand": np.random.randint(40, 250, 100),
        "transport_cost": np.random.randint(10, 100, 100),
        "lead_time": np.random.randint(1, 10, 100)
    })
    df["delay"] = df["lead_time"] + np.random.randint(0, 5, 100)
    df["risk"] = (df["delay"] > 8).astype(int)
    return df

# =========================
# LOAD DATA
# =========================
st.sidebar.header("📂 Data Input")
file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

df = load_data(file) if file else generate_data()

if df is None or df.empty:
    st.error("❌ Failed to load data")
    st.stop()

# =========================
# REQUIRED COLUMN CHECK
# =========================
required = ["inventory", "demand", "transport_cost", "lead_time"]
for col in required:
    if col not in df.columns:
        df[col] = np.random.randint(10, 100, len(df))

if "delay" not in df.columns:
    df["delay"] = df["lead_time"] + np.random.randint(0, 5, len(df))

if "risk" not in df.columns:
    df["risk"] = (df["delay"] > 8).astype(int)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# =========================
# PREPROCESSING
# =========================
df.fillna(0, inplace=True)

X = df[required]
y_delay = df["delay"]
y_risk = df["risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN MODELS (FAST)
# =========================
delay_model = DecisionTreeRegressor(max_depth=5)
delay_model.fit(X_scaled, y_delay)

risk_model = RandomForestClassifier(n_estimators=50)
risk_model.fit(X_scaled, y_risk)

demand_model = LinearRegression()
demand_model.fit(df[["inventory"]], df["demand"])

# =========================
# UI CONTROLS
# =========================
st.sidebar.header("⚙️ Simulation")

scenario = st.sidebar.selectbox("Scenario", [
    "Normal",
    "Flood (Delay)",
    "Demand Spike",
    "Transport Cost Increase"
])

demand_pct = st.sidebar.slider("Demand Increase %", 0, 100, 20)
delay_days = st.sidebar.slider("Extra Delay Days", 0, 10, 3)

# =========================
# SIMULATION
# =========================
df_sim = df.copy()

if scenario == "Flood (Delay)":
    df_sim["lead_time"] += delay_days

elif scenario == "Demand Spike":
    df_sim["demand"] *= (1 + demand_pct / 100)

elif scenario == "Transport Cost Increase":
    df_sim["transport_cost"] *= 1.5

# =========================
# PREDICTIONS
# =========================
X_sim = scaler.transform(df_sim[required])

df_sim["pred_delay"] = delay_model.predict(X_sim)
df_sim["pred_risk"] = risk_model.predict(X_sim)

# =========================
# DECISION ENGINE
# =========================
recommendations = []

if df_sim["pred_risk"].mean() > 0.5:
    recommendations.append("⚠️ Switch Supplier")
else:
    recommendations.append("✅ System Stable")

if df_sim["demand"].mean() > df["inventory"].mean():
    recommendations.append("📦 Increase Inventory")

if scenario == "Transport Cost Increase":
    recommendations.append("🚚 Optimize Route")

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
col3.metric("Cost Impact", f"{new_cost - base_cost:.2f}")

# =========================
# VISUALS (SAFE)
# =========================
st.subheader("Demand vs Inventory")
fig1 = px.scatter(df_sim, x="inventory", y="demand", color=df_sim["pred_risk"].astype(str))
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Cost Comparison")
cost_df = pd.DataFrame({
    "Stage": ["Before", "After"],
    "Cost": [base_cost, new_cost]
})
fig2 = px.bar(cost_df, x="Stage", y="Cost")
st.plotly_chart(fig2, use_container_width=True)

# =========================
# RECOMMENDATIONS
# =========================
st.header("🧠 AI Decisions")
for r in recommendations:
    st.success(r)

# =========================
# REPORT
# =========================
st.header("📄 Report")
st.write(f"""
Scenario: {scenario}

Average Delay: {df_sim['pred_delay'].mean():.2f}

Risk Level: {"High" if df_sim["pred_risk"].mean() > 0.5 else "Low"}

Cost Change: {new_cost - base_cost:.2f}
""")
