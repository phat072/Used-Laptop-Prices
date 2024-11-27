import json
import pandas as pd
import plotly.express as px
import streamlit as st

# PAGE SETUP
st.set_page_config(page_title="Product Dashboard", page_icon=":bar_chart:", layout="wide")
st.title("Product Sales Dashboard")
st.markdown("_Prototype v0.1_")

# Load JSON data
with st.sidebar:
    st.header("Upload Configuration")
    uploaded_file = st.file_uploader("Upload JSON file", type=["json"])

if uploaded_file is None:
    st.info("Please upload a file to continue.", icon="ℹ️")
    st.stop()

# Load data
@st.cache_data
def load_json(file):
    data = json.load(file)
    return pd.DataFrame(data)

df = load_json(uploaded_file)

with st.expander("Data Preview"):
    st.dataframe(df)

# Dashboard Visualizations
st.header("Dashboard Overview")

# Price distribution by condition
st.subheader("Price Distribution by Condition")
fig_condition = px.box(
    df,
    x="condition",
    y="price",
    title="Price Distribution by Condition",
    color="condition",
    labels={"price": "Price (VND)", "condition": "Condition"},
)
st.plotly_chart(fig_condition, use_container_width=True)

# Average price by brand
st.subheader("Average Price by Brand")
fig_brand = px.bar(
    df.groupby("brand")["price"].mean().reset_index(),
    x="brand",
    y="price",
    text_auto=".2s",
    title="Average Price by Brand",
    labels={"price": "Average Price (VND)", "brand": "Brand"},
    color="brand",
)
st.plotly_chart(fig_brand, use_container_width=True)

# Screen size distribution
st.subheader("Screen Size Distribution")
fig_screen = px.histogram(
    df,
    x="screen_size",
    nbins=10,
    title="Screen Size Distribution",
    labels={"screen_size": "Screen Size (inches)"},
)
st.plotly_chart(fig_screen, use_container_width=True)

# RAM distribution
st.subheader("RAM Distribution")
fig_ram = px.pie(
    df,
    names="ram",
    title="RAM Distribution",
    labels={"ram": "RAM (GB)"},
    hole=0.4,
)
st.plotly_chart(fig_ram, use_container_width=True)
