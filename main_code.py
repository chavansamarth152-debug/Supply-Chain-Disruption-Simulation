import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="AI Supply Chain DSS", layout="wide")
st.title("🚀 AI Supply Chain Disruption System")

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

st.write("### 📊 Data Preview")
st.write(df)

# -------------------------------
# SAFE COLUMN HANDLING
# -------------------------------
def col(name, default):
    return df[name] if name in df.columns else pd.Series([default]*len(df))

df["Reliability"] = col("Reliability", 0.8)
df["Inventory"] = col("Inventory", 300)
df["Demand"] = col("Demand", 300)
df["Transport_Cost"] = col("Transport_Cost", 50)
df["Lead_Time"] = col("Lead_Time", 5)
df["Product_Cost"] = col("Product_Cost", 20)

# Create ID if missing
if "Supplier_ID" not in df.columns:
    df["Supplier_ID"] = ["S"+str(i) for i in range(len(df))]

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Simulation Controls")

scenario = st.sidebar.selectbox("Scenario", [
    "None",
    "Demand Spike",
    "Flood Delay",
    "Transport Cost Increase"
])

demand_inc = st.sidebar.slider("Demand Increase %", 0, 100, 20)
delay_days = st.sidebar.slider("Delay Days", 0, 10, 2)

# -------------------------------
# SIMULATION
# -------------------------------
original = df.copy()
sim = df.copy()

if scenario == "Demand Spike":
    sim["Demand"] = original["Demand"] * (1 + demand_inc / 100)

if scenario == "Flood Delay":
    sim["Lead_Time"] = original["Lead_Time"] + delay_days

if scenario == "Transport Cost Increase":
    sim["Transport_Cost"] = original["Transport_Cost"] * 1.5

# -------------------------------
# ML MODEL
# -------------------------------
rmse = 0
try:
    if len(df) > 3:
        X = original[["Inventory","Lead_Time","Transport_Cost"]]
        y = original["Demand"]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = LinearRegression()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
except:
    pass

# -------------------------------
# COST
# -------------------------------
orig_cost = (original["Transport_Cost"] + original["Product_Cost"]).sum()
new_cost = (sim["Transport_Cost"] + sim["Product_Cost"]).sum()

# -------------------------------
# METRICS
# -------------------------------
st.subheader("📊 Key Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Original Cost", round(orig_cost,2))
c2.metric("New Cost", round(new_cost,2))
c3.metric("RMSE", round(rmse,2))

# -------------------------------
# HISTOGRAM
# -------------------------------
st.subheader("📊 Demand Distribution (Histogram)")
st.plotly_chart(px.histogram(sim, x="Demand"))

# -------------------------------
# SCATTER PLOT (FIXED)
# -------------------------------
st.subheader("🔥 Risk Scatter Plot")

sim["Risk"] = (1 - sim["Reliability"]) * sim["Lead_Time"]

st.plotly_chart(
    px.scatter(
        sim,
        x="Reliability",
        y="Lead_Time",
        size="Inventory",
        color="Risk"
    )
)

# -------------------------------
# LINE GRAPH
# -------------------------------
st.subheader("📈 Demand Change (Line Graph)")

line_df = pd.DataFrame({
    "Original": original["Demand"],
    "Simulated": sim["Demand"]
})

st.plotly_chart(px.line(line_df))

# -------------------------------
# PIE CHART
# -------------------------------
st.subheader("🥧 Cost Distribution")

cost_df = pd.DataFrame({
    "Type": ["Transport", "Product"],
    "Cost": [sim["Transport_Cost"].sum(), sim["Product_Cost"].sum()]
})

st.plotly_chart(px.pie(cost_df, names="Type", values="Cost"))

# -------------------------------
# BAR COMPARISON
# -------------------------------
st.subheader("📊 Before vs After Demand")

bar_df = pd.DataFrame({
    "Supplier": sim["Supplier_ID"],
    "Original": original["Demand"],
    "Simulated": sim["Demand"]
})

st.plotly_chart(px.bar(bar_df, x="Supplier", y=["Original","Simulated"], barmode="group"))

# -------------------------------
# RECOMMENDATIONS
# -------------------------------
st.subheader("🧠 AI Recommendations")

rec = []

if (sim["Inventory"] < sim["Demand"]).any():
    rec.append("Increase Inventory")

if (sim["Lead_Time"] > original["Lead_Time"]).any():
    rec.append("Reduce Delay / Use Faster Routes")

if (sim["Transport_Cost"] > original["Transport_Cost"]).any():
    rec.append("Optimize Transport Cost")

if not rec:
    rec.append("System Stable")

for r in rec:
    st.success(r)

# -------------------------------
# DOWNLOAD
# -------------------------------
st.download_button("⬇ Download Report", sim.to_csv(index=False), "report.csv")
