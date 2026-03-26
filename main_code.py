import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(layout="wide")
st.title("🚀 AI Supply Chain DSS with Monte Carlo Simulation")

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

if "Supplier_ID" not in df.columns:
    df["Supplier_ID"] = ["S"+str(i) for i in range(len(df))]

# -------------------------------
# ADD VARIATION (IMPORTANT FIX)
# -------------------------------
df["Demand"] = df["Demand"] + np.random.randint(-50, 50, size=len(df))
df["Lead_Time"] = df["Lead_Time"] + np.random.randint(-2, 2, size=len(df))

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Simulation")

scenario = st.sidebar.selectbox("Scenario", [
    "None", "Demand Spike", "Delay", "Cost Increase"
])

demand_inc = st.sidebar.slider("Demand % Increase", 0, 100, 20)
delay = st.sidebar.slider("Delay Days", 0, 10, 2)

# -------------------------------
# SIMULATION
# -------------------------------
original = df.copy()
sim = df.copy()

if scenario == "Demand Spike":
    sim["Demand"] = sim["Demand"] * (1 + demand_inc/100)

if scenario == "Delay":
    sim["Lead_Time"] = sim["Lead_Time"] + delay

if scenario == "Cost Increase":
    sim["Transport_Cost"] *= 1.5

# -------------------------------
# MONTE CARLO SIMULATION
# -------------------------------
st.subheader("🎲 Monte Carlo Simulation")

runs = 100
results = []

for i in range(runs):
    temp = sim.copy()
    temp["Demand"] = temp["Demand"] * np.random.uniform(0.9, 1.2)
    total_cost = (temp["Transport_Cost"] + temp["Product_Cost"]).sum()
    results.append(total_cost)

mc_df = pd.DataFrame({"Cost": results})

st.plotly_chart(px.histogram(mc_df, x="Cost", nbins=20))

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
# GRAPHS
# -------------------------------

st.subheader("📊 Histogram (Demand)")
st.plotly_chart(px.histogram(sim, x="Demand"))

st.subheader("📈 Line Graph (Demand Trend)")
st.plotly_chart(px.line(sim, y="Demand"))

st.subheader("🔥 Scatter Plot (Risk)")
sim["Risk"] = (1 - sim["Reliability"]) * sim["Lead_Time"]

st.plotly_chart(px.scatter(sim,
                           x="Reliability",
                           y="Lead_Time",
                           size="Inventory",
                           color="Risk"))

st.subheader("🥧 Pie Chart (Cost Split)")
cost_df = pd.DataFrame({
    "Type": ["Transport","Product"],
    "Value": [sim["Transport_Cost"].sum(), sim["Product_Cost"].sum()]
})
st.plotly_chart(px.pie(cost_df, names="Type", values="Value"))

# -------------------------------
# DECISION ENGINE (ADVANCED TEXT)
# -------------------------------
st.subheader("🧠 AI Recommendations")

recommendation_text = ""

if sim["Demand"].mean() > original["Demand"].mean():
    recommendation_text += "• Demand is increasing → Increase inventory planning.\n"

if sim["Lead_Time"].mean() > original["Lead_Time"].mean():
    recommendation_text += "• Delivery delays detected → Use faster logistics or backup suppliers.\n"

if sim["Transport_Cost"].mean() > original["Transport_Cost"].mean():
    recommendation_text += "• Cost rising → Optimize routes and reduce dependency on expensive suppliers.\n"

if sim["Reliability"].mean() < 0.75:
    recommendation_text += "• Supplier reliability is low → Switch to more reliable vendors.\n"

if recommendation_text == "":
    recommendation_text = "• System stable. No major risks detected."

st.text_area("AI Decision Insights", recommendation_text, height=200)

# -------------------------------
# METRICS
# -------------------------------
st.subheader("📊 Key Metrics")

orig_cost = (original["Transport_Cost"] + original["Product_Cost"]).sum()
new_cost = (sim["Transport_Cost"] + sim["Product_Cost"]).sum()

c1, c2, c3 = st.columns(3)
c1.metric("Original Cost", round(orig_cost,2))
c2.metric("New Cost", round(new_cost,2))
c3.metric("Model RMSE", round(rmse,2))

# -------------------------------
# DOWNLOAD
# -------------------------------
st.download_button("Download Report", sim.to_csv(index=False), "report.csv")
