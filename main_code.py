import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("🚀 AI Supply Chain DSS (Final Version)")

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
# SAFE COLUMNS
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
# ADD VARIATION (KEY FIX)
# -------------------------------
df["Demand"] += np.random.randint(-50, 50, len(df))
df["Lead_Time"] += np.random.randint(-2, 3, len(df))

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Simulation")

scenario = st.sidebar.selectbox("Scenario", [
    "None", "Demand Spike", "Delay", "Cost Increase"
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

if scenario == "Delay":
    sim["Lead_Time"] = sim["Lead_Time"] + delay

if scenario == "Cost Increase":
    sim["Transport_Cost"] = sim["Transport_Cost"] * 1.5

# -------------------------------
# DEMAND TREND (FIXED)
# -------------------------------
st.subheader("📈 Demand Trend")

trend_df = pd.DataFrame({
    "Original": original["Demand"],
    "Simulated": sim["Demand"]
})

st.plotly_chart(px.line(trend_df))

# -------------------------------
# MONTE CARLO (FIXED PROPERLY)
# -------------------------------
st.subheader("🎲 Monte Carlo Simulation")

runs = 500
results = []

for i in range(runs):
    temp = sim.copy()

    temp["Demand"] *= np.random.normal(1.0, 0.2, len(temp))
    temp["Lead_Time"] += np.random.randint(-2, 3, len(temp))
    temp["Transport_Cost"] *= np.random.uniform(0.8, 1.5)

    cost = (
        temp["Transport_Cost"] +
        temp["Product_Cost"] +
        (temp["Demand"] * 0.05)
    ).sum()

    results.append(cost)

mc_df = pd.DataFrame({"Cost": results})

st.plotly_chart(px.histogram(mc_df, x="Cost", nbins=30))

st.write("Average Cost:", round(mc_df["Cost"].mean(),2))
st.write("Max Cost:", round(mc_df["Cost"].max(),2))
st.write("Min Cost:", round(mc_df["Cost"].min(),2))

# -------------------------------
# GRAPHS
# -------------------------------
st.subheader("📊 Histogram")
st.plotly_chart(px.histogram(sim, x="Demand"))

st.subheader("🔥 Scatter Plot")
sim["Risk"] = (1 - sim["Reliability"]) * sim["Lead_Time"]
st.plotly_chart(px.scatter(sim, x="Reliability", y="Lead_Time",
                           size="Inventory", color="Risk"))

st.subheader("🥧 Pie Chart")
cost_df = pd.DataFrame({
    "Type":["Transport","Product"],
    "Value":[sim["Transport_Cost"].sum(), sim["Product_Cost"].sum()]
})
st.plotly_chart(px.pie(cost_df, names="Type", values="Value"))

# -------------------------------
# AI RECOMMENDATION (STRONG TEXT)
# -------------------------------
st.subheader("🧠 AI Insights")

text = ""

if sim["Demand"].mean() > original["Demand"].mean():
    text += "Demand surge detected → Increase inventory and safety stock.\n"

if sim["Lead_Time"].mean() > original["Lead_Time"].mean():
    text += "Lead time increased → Use faster transportation or alternate suppliers.\n"

if sim["Transport_Cost"].mean() > original["Transport_Cost"].mean():
    text += "Transport cost rise → Optimize routes and reduce dependency.\n"

if sim["Reliability"].mean() < 0.75:
    text += "Supplier reliability risk → Diversify supplier base.\n"

if text == "":
    text = "System stable with no major disruptions."

st.text_area("Decision Support Output", text, height=200)

# -------------------------------
# METRICS
# -------------------------------
st.subheader("📊 Metrics")

orig_cost = (original["Transport_Cost"] + original["Product_Cost"]).sum()
new_cost = (sim["Transport_Cost"] + sim["Product_Cost"]).sum()

c1,c2 = st.columns(2)
c1.metric("Original Cost", round(orig_cost,2))
c2.metric("New Cost", round(new_cost,2))

# -------------------------------
# PDF
# -------------------------------
def create_pdf(text):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Supply Chain DSS Report", styles['Title']))
    story.append(Spacer(1,12))
    story.append(Paragraph(text, styles['Normal']))

    doc.build(story)

create_pdf(text)

with open("report.pdf", "rb") as f:
    st.download_button("📄 Download PDF", f, "report.pdf")

# -------------------------------
# DOWNLOAD CSV
# -------------------------------
st.download_button("⬇ Download Data", sim.to_csv(index=False), "data.csv")
