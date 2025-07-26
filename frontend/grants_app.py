import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8000/grants"  # adjust if deployed

st.set_page_config(page_title="Grants Viewer", layout="wide")
st.title("📊 Government Grants Viewer")

# Fetch data from FastAPI backend
@st.cache_data(ttl=60)
def fetch_data():
    res = requests.get(API_URL)
    return pd.DataFrame(res.json())

df = fetch_data()

# Sidebar filters
st.sidebar.header("🔍 Filter")

# --- Dropdown by predicted_label ---
if "predicted_label" in df.columns:
    categories = ["All"] + sorted(df["predicted_label"].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Categories", categories)

    if selected_category != "All":
        df = df[df["predicted_label"] == selected_category]

# --- Text search ---
text_filter = st.sidebar.text_input("Search in any column")
if text_filter:
    df = df[df.apply(lambda row: row.astype(str).str.contains(text_filter, case=False).any(), axis=1)]

# Display table
st.dataframe(df, use_container_width=True)
st.caption("Last updated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
