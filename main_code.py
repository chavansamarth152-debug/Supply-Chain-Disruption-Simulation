import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import networkx as nx
import seaborn as sns
import json
import io
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("📈 AI Supply Chain Forecast & Simulation")

# =========================
# FILE UPLOAD
# =========================
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file")
    st.stop()

required_cols = {'product_id', 'date', 'sales'}
if not required_cols.issubset(df_raw.columns):
    st.error("CSV must contain: product_id, date, sales")
    st.stop()

# =========================
# DATA PREP
# =========================
product_id = st.sidebar.selectbox("Select Product", df_raw['product_id'].unique())

df = df_raw[df_raw['product_id'] == product_id][['date', 'sales']]
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

forecast_days = st.sidebar.slider("Forecast Days", 7, 180, 30)

# =========================
# MODELS
# =========================
@st.cache_resource
def train_prophet(df):
    model = Prophet()
    model.fit(df)
    return model

@st.cache_resource
def train_xgb(df):
    df = df.set_index('ds').resample('D').sum().fillna(0)
    df['lag1'] = df['y'].shift(1)
    df.dropna(inplace=True)

    X = df[['lag1']]
    y = df['y']

    model = xgb.XGBRegressor()
    model.fit(X, y)
    return model, df

# =========================
# TRAIN
# =========================
with st.spinner("Training models..."):
    prophet_model = train_prophet(df.copy())
    xgb_model, df_xgb = train_xgb(df.copy())

# =========================
# FORECAST
# =========================
st.header("📊 Forecast")

# Prophet Forecast
future = prophet_model.make_future_dataframe(periods=forecast_days)
forecast = prophet_model.predict(future)

fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast"))
st.plotly_chart(fig, use_container_width=True)

# =========================
# SIMPLE METRICS
# =========================
true = df['y'].tail(20)
pred = forecast['yhat'].tail(20)

mae = mean_absolute_error(true, pred)
rmse = np.sqrt(mean_squared_error(true, pred))

st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")

# =========================
# SUPPLY CHAIN GRAPH
# =========================
st.header("🚧 Supply Chain Simulation")

default_graph = {
    "nodes": ["Factory", "Port", "Warehouse"],
    "edges": [
        ("Factory", "Port", 5),
        ("Port", "Warehouse", 3)
    ]
}

G = nx.DiGraph()
for n in default_graph["nodes"]:
    G.add_node(n)

for u, v, d in default_graph["edges"]:
    G.add_edge(u, v, transit_days=d)

# Delay sliders
for edge in G.edges:
    G.edges[edge]['transit_days'] = st.slider(
        f"{edge[0]} ➜ {edge[1]} delay",
        1, 20, G.edges[edge]['transit_days']
    )

# =========================
# HEATMAP
# =========================
def compute_matrix(G):
    nodes = list(G.nodes)
    matrix = pd.DataFrame(index=nodes, columns=nodes)

    for i in nodes:
        for j in nodes:
            if i != j:
                try:
                    path = nx.shortest_path(G, i, j, weight='transit_days')
                    delay = sum(G[u][v]['transit_days'] for u, v in zip(path[:-1], path[1:]))
                    matrix.loc[i, j] = delay
                except:
                    matrix.loc[i, j] = None
    return matrix

heat = compute_matrix(G)

st.subheader("Delay Heatmap")
fig, ax = plt.subplots()
sns.heatmap(heat.astype(float), annot=True, cmap="Reds", ax=ax)
st.pyplot(fig)

# =========================
# COST SIMULATION
# =========================
st.header("💰 Cost Simulation")

distance = st.number_input("Distance (km)", 0.0)
cost_km = st.number_input("Cost per km", 0.0)
qty = st.number_input("Quantity", 0.0)

cost = distance * cost_km * qty
st.metric("Total Cost", f"${cost:.2f}")

# =========================
# MONTE CARLO
# =========================
st.header("🎲 Risk Simulation")

runs = st.slider("Simulation Runs", 100, 500, 200)
prob = st.slider("Disruption Probability", 0.0, 1.0, 0.3)

delays = []

for _ in range(runs):
    total = 0
    for edge in G.edges:
        d = G.edges[edge]['transit_days']
        if np.random.rand() < prob:
            d += np.random.randint(1, 5)
        total += d
    delays.append(total)

fig2, ax2 = plt.subplots()
sns.histplot(delays, bins=20, kde=True, ax=ax2)
st.pyplot(fig2)

# =========================
# DOWNLOAD
# =========================
st.header("⬇️ Download Forecast")

download_df = forecast[['ds', 'yhat']]
csv = download_df.to_csv(index=False)

st.download_button("Download CSV", csv, "forecast.csv", "text/csv")
