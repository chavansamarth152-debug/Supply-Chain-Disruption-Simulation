import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(layout="wide")
st.title("🌍 AI Supply Chain Disruption Simulation System")

# -------------------------------
# LOAD CSV (FAST)
# -------------------------------
file = st.file_uploader("Upload CSV", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if file:
    df = load_data(file)
else:
    df = pd.DataFrame({
        "supplier_region": ["Asia-Pacific","Europe","North America","South America"],
        "disruption_type": ["Flood","Strike","Cyber","War"],
        "industry": ["Electronics","Auto","Pharma","Retail"],
        "disruption_severity": [3,4,2,5]
    })

st.write("### 📊 Data Preview")
st.write(df.head())

# -------------------------------
# REGION MAPPING
# -------------------------------
region_map = {
    "Asia-Pacific": ("India", 20, 78),
    "Europe": ("Germany", 51, 10),
    "North America": ("USA", 37, -95),
    "South America": ("Brazil", -14, -51),
    "Africa": ("Nigeria", 9, 8),
    "Middle East": ("UAE", 24, 54)
}

df["Lat"] = df["supplier_region"].map(lambda x: region_map.get(x, ("",0,0))[1])
df["Lon"] = df["supplier_region"].map(lambda x: region_map.get(x, ("",0,0))[2])

# -------------------------------
# 🔥 PERFORMANCE BOOST
# -------------------------------
# Take sample if large dataset
if len(df) > 5000:
    df_sample = df.sample(2000)
else:
    df_sample = df

# Aggregate for map speed
map_df = df_sample.groupby(["supplier_region","Lat","Lon"], as_index=False).agg({
    "disruption_severity":"mean"
})

# -------------------------------
# RISK LEVELS (NEW FEATURE)
# -------------------------------
def risk_level(x):
    if x < 2:
        return "Low"
    elif x < 4:
        return "Medium"
    else:
        return "High"

map_df["Risk_Level"] = map_df["disruption_severity"].apply(risk_level)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Simulation")

scenario = st.sidebar.selectbox("Scenario", [
    "None","High Disruption","Extreme Risk"
])

if scenario == "High Disruption":
    map_df["disruption_severity"] *= 1.5

if scenario == "Extreme Risk":
    map_df["disruption_severity"] *= 2

# -------------------------------
# 🌍 FAST STATIC MAP
# -------------------------------
st.subheader("🌍 Global Disruption Map")

fig = px.scatter_geo(
    map_df,
    lat="Lat",
    lon="Lon",
    color="Risk_Level",
    size="disruption_severity",
    hover_name="supplier_region",
    color_discrete_map={
        "Low": "green",
        "Medium": "orange",
        "High": "red"
    }
)

fig.update_layout(
    geo=dict(showland=True),
    title="Supply Chain Disruption Risk Map"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 📊 GRAPHS
# -------------------------------
st.subheader("📊 Severity Distribution")
st.plotly_chart(px.histogram(df_sample, x="disruption_severity"))

st.subheader("📈 Trend")
st.plotly_chart(px.line(df_sample["disruption_severity"]))

st.subheader("🔥 Risk Scatter")
st.plotly_chart(px.scatter(
    df_sample,
    x="disruption_severity",
    y=np.arange(len(df_sample)),
    color="disruption_severity"
))

st.subheader("🥧 Industry Impact")
if "industry" in df.columns:
    st.plotly_chart(px.pie(df_sample, names="industry", values="disruption_severity"))

# -------------------------------
# 🎲 MONTE CARLO (FAST)
# -------------------------------
st.subheader("🎲 Monte Carlo Simulation")

runs = 300
results = []

for i in range(runs):
    temp = df_sample.copy()
    temp["disruption_severity"] *= np.random.normal(1, 0.3, len(temp))
    cost = (temp["disruption_severity"] * 100).sum()
    results.append(cost)

mc_df = pd.DataFrame({"Cost": results})

st.plotly_chart(px.histogram(mc_df, x="Cost", nbins=25))

# -------------------------------
# 🧠 AI RECOMMENDATION
# -------------------------------
st.subheader("🧠 AI Insights")

text = ""

if df_sample["disruption_severity"].mean() > 3:
    text += "• High disruption risk → Activate contingency planning\n"

if "Cyber" in df_sample.get("disruption_type", []).values:
    text += "• Cyber risk detected → Strengthen cybersecurity\n"

if "Strike" in df_sample.get("disruption_type", []).values:
    text += "• Labor issues → Prepare backup workforce\n"

if text == "":
    text = "• System stable"

st.text_area("Decision Support", text, height=200)

# -------------------------------
# 📄 PDF
# -------------------------------
def create_pdf(text):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Supply Chain Report", styles['Title']))
    story.append(Spacer(1,12))
    story.append(Paragraph(text, styles['Normal']))

    doc.build(story)

create_pdf(text)

with open("report.pdf", "rb") as f:
    st.download_button("📄 Download PDF", f, "report.pdf")

# -------------------------------
# TABLE
# -------------------------------
st.subheader("📋 Data Table")
st.dataframe(df_sample)
