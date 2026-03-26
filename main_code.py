import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(layout="wide")
st.title("🌍 AI Supply Chain Disruption Simulation System")

# -------------------------------
# LOAD CSV
# -------------------------------
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
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
# REGION → COUNTRY + COORDS
# -------------------------------
region_map = {
    "Asia-Pacific": ("India", 20, 78),
    "Europe": ("Germany", 51, 10),
    "North America": ("USA", 37, -95),
    "South America": ("Brazil", -14, -51),
    "Africa": ("Nigeria", 9, 8),
    "Middle East": ("UAE", 24, 54)
}

df["Country"] = df["supplier_region"].map(lambda x: region_map.get(x, ("Unknown",0,0))[0])
df["Lat"] = df["supplier_region"].map(lambda x: region_map.get(x, ("",0,0))[1])
df["Lon"] = df["supplier_region"].map(lambda x: region_map.get(x, ("",0,0))[2])

# Add fake timeline for animation
df["Step"] = np.arange(len(df))

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Simulation Controls")

scenario = st.sidebar.selectbox("Scenario", [
    "None","High Disruption","Extreme Risk"
])

if scenario == "High Disruption":
    df["disruption_severity"] *= 1.5

if scenario == "Extreme Risk":
    df["disruption_severity"] *= 2

# -------------------------------
# 🌍 ANIMATED MAP
# -------------------------------
st.subheader("🌍 Global Disruption Map (Animated)")

fig = px.scatter_geo(
    df,
    lat="Lat",
    lon="Lon",
    color="disruption_severity",
    size="disruption_severity",
    hover_name="disruption_type",
    hover_data=["industry","supplier_region"],
    animation_frame="Step",
    color_continuous_scale="Reds"
)

fig.update_layout(
    geo=dict(showland=True),
    title="Global Supply Chain Disruptions"
)

st.plotly_chart(fig)

# -------------------------------
# 📊 HISTOGRAM
# -------------------------------
st.subheader("📊 Severity Distribution")
st.plotly_chart(px.histogram(df, x="disruption_severity"))

# -------------------------------
# 📈 LINE GRAPH
# -------------------------------
st.subheader("📈 Disruption Trend")
st.plotly_chart(px.line(df, y="disruption_severity"))

# -------------------------------
# 🔥 SCATTER
# -------------------------------
st.subheader("🔥 Risk Analysis")
st.plotly_chart(px.scatter(
    df,
    x="disruption_severity",
    y="Step",
    color="disruption_severity",
    size="disruption_severity"
))

# -------------------------------
# 🥧 PIE
# -------------------------------
st.subheader("🥧 Industry Impact")
st.plotly_chart(px.pie(df, names="industry", values="disruption_severity"))

# -------------------------------
# 🎲 MONTE CARLO
# -------------------------------
st.subheader("🎲 Monte Carlo Simulation")

runs = 500
results = []

for i in range(runs):
    temp = df.copy()
    temp["disruption_severity"] *= np.random.normal(1, 0.3, len(temp))
    cost = (temp["disruption_severity"] * 100).sum()
    results.append(cost)

mc_df = pd.DataFrame({"Cost": results})

st.plotly_chart(px.histogram(mc_df, x="Cost", nbins=30))

st.write("Average Risk Cost:", round(mc_df["Cost"].mean(),2))
st.write("Max Risk:", round(mc_df["Cost"].max(),2))

# -------------------------------
# 🧠 AI RECOMMENDATION
# -------------------------------
st.subheader("🧠 AI Decision Insights")

text = ""

if df["disruption_severity"].mean() > 3:
    text += "• High disruption risk detected → Activate contingency plans\n"

if "Cyber" in df["disruption_type"].values:
    text += "• Cyber threats present → Strengthen cybersecurity\n"

if "Strike" in df["disruption_type"].values:
    text += "• Labor disruption → Identify alternate workforce\n"

if text == "":
    text = "• Supply chain stable"

st.text_area("AI Output", text, height=200)

# -------------------------------
# 📄 PDF REPORT
# -------------------------------
def create_pdf(text):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Supply Chain Disruption Report", styles['Title']))
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
st.dataframe(df)
