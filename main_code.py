import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk

st.set_page_config(layout="wide")
st.title("🌍 Universal Supply Chain DSS (Any CSV Supported)")

# -------------------------------
# LOAD DATA
# -------------------------------
file = st.file_uploader("Upload ANY CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    df = pd.DataFrame({
        "Name": ["A","B","C"],
        "Demand": [300,400,350],
        "Inventory": [500,200,300],
        "Reliability": [0.9,0.7,0.8],
        "Lat": [20,25,30],
        "Lon": [70,75,80]
    })

st.write("### 📊 Data Preview")
st.write(df.head())

# -------------------------------
# AUTO COLUMN DETECTION
# -------------------------------
def find_col(possible_names):
    for col in df.columns:
        for name in possible_names:
            if name.lower() in col.lower():
                return col
    return None

demand_col = find_col(["demand","sales"])
inventory_col = find_col(["inventory","stock"])
reliability_col = find_col(["reliability","score"])
lat_col = find_col(["lat","latitude"])
lon_col = find_col(["lon","lng","longitude"])
name_col = df.columns[0]

# fallback safe values
df["Demand"] = df[demand_col] if demand_col else np.random.randint(200,500,len(df))
df["Inventory"] = df[inventory_col] if inventory_col else np.random.randint(200,500,len(df))
df["Reliability"] = df[reliability_col] if reliability_col else np.random.uniform(0.6,0.95,len(df))

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Simulation")

scenario = st.sidebar.selectbox("Scenario", [
    "None","Demand Spike","Delay","Disruption"
])

demand_inc = st.sidebar.slider("Demand Increase %", 0, 100, 20)

# -------------------------------
# SIMULATION
# -------------------------------
original = df.copy()
sim = df.copy()

if scenario == "Demand Spike":
    sim["Demand"] *= (1 + demand_inc/100)

if scenario == "Disruption":
    sim["Reliability"] *= 0.5

# -------------------------------
# MAP (SAFE)
# -------------------------------
st.subheader("🌍 Supply Chain Map")

if lat_col and lon_col:
    df["Lat"] = df[lat_col]
    df["Lon"] = df[lon_col]

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=sim,
        get_position='[Lon, Lat]',
        get_radius=50000,
        get_color='[0,150,255]'
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1)
    ))
else:
    st.warning("No location data found → Map disabled")

# -------------------------------
# GRAPHS
# -------------------------------
st.subheader("📊 Histogram")
st.plotly_chart(px.histogram(sim, x="Demand"))

st.subheader("📈 Line Graph")
st.plotly_chart(px.line(sim["Demand"]))

st.subheader("🔥 Scatter")
sim["Risk"] = (1 - sim["Reliability"]) * sim["Demand"]
st.plotly_chart(px.scatter(sim, x="Reliability", y="Demand", color="Risk"))

st.subheader("🥧 Pie Chart")
pie_df = pd.DataFrame({
    "Type":["Demand","Inventory"],
    "Value":[sim["Demand"].sum(), sim["Inventory"].sum()]
})
st.plotly_chart(px.pie(pie_df, names="Type", values="Value"))

# -------------------------------
# MONTE CARLO
# -------------------------------
st.subheader("🎲 Monte Carlo Simulation")

runs = 300
costs = []

for i in range(runs):
    temp = sim.copy()
    temp["Demand"] *= np.random.normal(1,0.2,len(temp))
    temp["Inventory"] *= np.random.uniform(0.8,1.2)

    cost = (temp["Demand"] * 0.1).sum()
    costs.append(cost)

mc_df = pd.DataFrame({"Cost": costs})

st.plotly_chart(px.histogram(mc_df, x="Cost"))

# -------------------------------
# AI RECOMMENDATIONS
# -------------------------------
st.subheader("🧠 AI Decision Support")

text = ""

if sim["Demand"].mean() > original["Demand"].mean():
    text += "• Demand increasing → Increase production\n"

if sim["Reliability"].mean() < 0.75:
    text += "• Supplier risk → Diversify suppliers\n"

if sim["Inventory"].mean() < sim["Demand"].mean():
    text += "• Inventory shortage → Restock\n"

if text == "":
    text = "• System stable"

st.text_area("AI Insights", text, height=150)

# -------------------------------
# TABLE
# -------------------------------
st.subheader("📋 Data Table")
st.dataframe(sim)
