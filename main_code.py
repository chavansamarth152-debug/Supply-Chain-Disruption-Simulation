import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🌍 Supply Chain Disruption Simulation System")

# -------------------------------
# LOAD DATA
# -------------------------------
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    df = pd.DataFrame({
        "Name": ["Berlin","Munich","Paris","Rome"],
        "Type": ["Supplier","Customer","Customer","Supplier"],
        "Lat": [52.52,48.13,48.85,41.90],
        "Lon": [13.40,11.58,2.35,12.49],
        "Demand": [300,400,350,320],
        "Inventory": [500,300,200,450],
        "Reliability": [0.9,0.7,0.8,0.6]
    })

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Simulation Controls")

scenario = st.sidebar.selectbox("Scenario", [
    "None","Demand Spike","Delay","Disruption"
])

demand_inc = st.sidebar.slider("Demand Increase %", 0, 100, 20)

# -------------------------------
# SIMULATION
# -------------------------------
sim = df.copy()

if scenario == "Demand Spike":
    sim["Demand"] *= (1 + demand_inc/100)

if scenario == "Disruption":
    sim["Reliability"] *= 0.5

# -------------------------------
# MAP VIEW (MAIN UI)
# -------------------------------
st.subheader("📍 Supply Chain Network Map")

layer = pdk.Layer(
    "ScatterplotLayer",
    data=sim,
    get_position='[Lon, Lat]',
    get_color='[200, 30, 0, 160]',
    get_radius=50000,
)

view_state = pdk.ViewState(
    latitude=50,
    longitude=10,
    zoom=4,
)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
))

# -------------------------------
# NETWORK INSIGHTS
# -------------------------------
st.subheader("📊 Supply Chain Overview")

col1, col2 = st.columns(2)

col1.write("### Demand Distribution")
col1.plotly_chart(px.bar(sim, x="Name", y="Demand"))

col2.write("### Inventory Levels")
col2.plotly_chart(px.bar(sim, x="Name", y="Inventory"))

# -------------------------------
# RISK ANALYSIS
# -------------------------------
st.subheader("🔥 Risk Analysis")

sim["Risk"] = (1 - sim["Reliability"]) * sim["Demand"]

st.plotly_chart(px.scatter(
    sim,
    x="Reliability",
    y="Demand",
    size="Inventory",
    color="Risk",
    hover_name="Name"
))

# -------------------------------
# MONTE CARLO
# -------------------------------
st.subheader("🎲 Monte Carlo Simulation")

runs = 300
costs = []

for i in range(runs):
    temp = sim.copy()
    temp["Demand"] *= np.random.normal(1, 0.2, len(temp))
    temp["Inventory"] *= np.random.uniform(0.8, 1.2)

    cost = (temp["Demand"] * 0.1).sum()
    costs.append(cost)

mc_df = pd.DataFrame({"Cost": costs})

st.plotly_chart(px.histogram(mc_df, x="Cost"))

# -------------------------------
# AI DECISION ENGINE
# -------------------------------
st.subheader("🧠 AI Recommendations")

text = ""

if sim["Demand"].mean() > df["Demand"].mean():
    text += "• Demand surge → Increase inventory\n"

if sim["Reliability"].mean() < 0.75:
    text += "• Supplier disruption → Switch suppliers\n"

if sim["Inventory"].mean() < sim["Demand"].mean():
    text += "• Inventory shortage → Restock urgently\n"

if text == "":
    text = "• Supply chain stable"

st.text_area("Decision Output", text, height=150)

# -------------------------------
# TABLE VIEW (LIKE YOUR UI)
# -------------------------------
st.subheader("📋 Supply Chain Table")

st.dataframe(sim)
