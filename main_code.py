import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="Supply Chain AI DSS", layout="wide")

st.title("🚀 AI Supply Chain Disruption Simulator")
st.markdown("### Professional Decision Support System")

# -------------------------------
# LOAD DATA
# -------------------------------
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    df = pd.DataFrame({
        "Supplier_ID": ["S1","S2","S3","S4"],
        "Reliability": [0.9,0.7,0.8,0.6],
        "Inventory": [500,300,400,200],
        "Demand": [450,400,420,350],
        "Transport_Cost": [50,60,55,70],
        "Lead_Time": [5,7,6,8],
        "Product_Cost": [20,18,22,25]
    })

# SAFE COLUMN HANDLING
def safe(col, val):
    return df[col] if col in df.columns else pd.Series([val]*len(df))

df["Reliability"] = safe("Reliability", 0.8)
df["Inventory"] = safe("Inventory", 300)
df["Demand"] = safe("Demand", 300)
df["Transport_Cost"] = safe("Transport_Cost", 50)
df["Lead_Time"] = safe("Lead_Time", 5)
df["Product_Cost"] = safe("Product_Cost", 20)

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("⚙️ Simulation Control")

scenario = st.sidebar.selectbox("Scenario", [
    "None",
    "Flood",
    "Demand Spike",
    "Transport Strike"
])

demand_inc = st.sidebar.slider("Demand Increase %", 0, 100, 20)
delay = st.sidebar.slider("Delay Days", 0, 10, 2)

# -------------------------------
# SIMULATION
# -------------------------------
original = df.copy()
sim = df.copy()

if scenario == "Demand Spike":
    sim["Demand"] = sim["Demand"] * (1 + demand_inc/100)

if scenario == "Flood":
    sim["Lead_Time"] = sim["Lead_Time"] + delay

if scenario == "Transport Strike":
    sim["Transport_Cost"] = sim["Transport_Cost"] * 1.5

# -------------------------------
# ML MODEL (DEMAND)
# -------------------------------
rmse = 0

try:
    X = original[["Inventory","Lead_Time","Transport_Cost"]]
    y = original["Demand"]

    if len(df) > 3:
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = LinearRegression()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
except:
    pass

# -------------------------------
# COST ANALYSIS
# -------------------------------
orig_cost = (original["Transport_Cost"] + original["Product_Cost"]).sum()
new_cost = (sim["Transport_Cost"] + sim["Product_Cost"]).sum()

# -------------------------------
# DECISION ENGINE (SMART)
# -------------------------------
recommend = []

if (sim["Inventory"] < sim["Demand"]).any():
    recommend.append("⚠ Increase Inventory Immediately")

if (sim["Lead_Time"] > original["Lead_Time"]).any():
    recommend.append("🚚 Use Faster Transport Route")

if (sim["Transport_Cost"] > original["Transport_Cost"]).any():
    recommend.append("💰 Optimize Logistics Cost")

if (sim["Reliability"] < 0.7).any():
    recommend.append("🔄 Switch to High-Reliability Supplier")

if not recommend:
    recommend.append("✅ System Operating Optimally")

# -------------------------------
# KPI DASHBOARD
# -------------------------------
st.subheader("📊 Key Metrics")

c1,c2,c3 = st.columns(3)

c1.metric("Original Cost", round(orig_cost,2))
c2.metric("New Cost", round(new_cost,2))
c3.metric("RMSE", round(rmse,2))

# -------------------------------
# GRAPH 1: DEMAND COMPARISON
# -------------------------------
st.subheader("📈 Demand Impact (Before vs After)")

fig = go.Figure()
fig.add_trace(go.Bar(name="Original", x=original.index, y=original["Demand"]))
fig.add_trace(go.Bar(name="After Simulation", x=sim.index, y=sim["Demand"]))

fig.update_layout(barmode='group')
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# GRAPH 2: LEAD TIME CHANGE
# -------------------------------
st.subheader("⏱ Lead Time Change")

fig2 = px.line(pd.DataFrame({
    "Original": original["Lead_Time"],
    "Simulated": sim["Lead_Time"]
}))
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# GRAPH 3: COST IMPACT
# -------------------------------
st.subheader("💰 Cost Comparison")

fig3 = px.bar(x=["Original","Simulated"], y=[orig_cost,new_cost])
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# GRAPH 4: RISK HEATMAP STYLE
# -------------------------------
st.subheader("🔥 Risk Visualization")

sim["Risk"] = (1 - sim["Reliability"]) * sim["Lead_Time"]

fig4 = px.scatter(sim, x="Reliability", y="Lead_Time",
                  size="Inventory", color="Risk",
                  hover_name="Supplier_ID")
st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# RECOMMENDATIONS
# -------------------------------
st.subheader("🧠 AI Recommendations")

for r in recommend:
    st.success(r)

# -------------------------------
# DOWNLOAD REPORT
# -------------------------------
st.download_button("⬇ Download Report",
                   sim.to_csv(index=False),
                   "report.csv")
