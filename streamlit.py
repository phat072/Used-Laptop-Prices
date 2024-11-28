import json
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Define numerical and categorical transformers
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Create value num_feature and cat_feature
num_feature = ["warranty", "screen_size", "ram", "hard_driver"]
cat_feature = ["brand", "model", "condition", "processor", "card", "made_in","hard_driver_kind"]

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

# List of model files for user to choose from
model_files = ["model_storage/decision_tree_model.joblib", "model_storage/ridge_model.joblib", "model_storage/random_forest_model.joblib"]

# Select prediction model
selected_model_file = st.selectbox("Select Prediction Model", model_files)

# Define preprocessor for transformations
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_feature),
    ('cat', cat_transformer, cat_feature)
])

X = df.drop(columns={"price"})
preprocessor.fit(X)

# Load model when the user selects it
@st.cache_resource
def load_model(model_file):
    model = joblib.load(model_file)
    return model

model = load_model(selected_model_file)

# Section for selecting product attributes
st.header("Select Product Attributes for Prediction")

# Convert 'condition' to categorical type before processing
df["condition"] = df["condition"].astype('category')

# Using Streamlit selectbox to get user inputs
select_brand = st.selectbox("Select Brand", df["brand"].unique())
select_model = st.selectbox("Select Model", df["model"].unique())
# select_condition = st.selectbox("Select Condition", df["condition"].unique())  # Ensure condition is categorical
select_condition = None
select_warranty = st.selectbox("Select Warranty", df["warranty"].unique())
select_screen_size = st.selectbox("Select Screen Size", df["screen_size"].unique())
select_processor = st.selectbox("Select Processor", df["processor"].unique())
select_ram = st.selectbox("Select RAM", df["ram"].unique())
select_card = st.selectbox("Select Card", df["card"].unique())
select_hard_driver = st.selectbox("Select Hard Driver", df["hard_driver"].unique())
select_made_in = st.selectbox("Select Made In", df["made_in"].unique())
select_hard_driver_kind = st.selectbox("Select Hard Driver Kind", df["hard_driver_kind"].unique())

# Create a list of selected values
selected_values = [
    select_brand, select_model, select_condition, select_warranty, 
    select_screen_size, select_processor, select_ram, select_card,
    select_hard_driver, select_made_in, select_hard_driver_kind
]

# Ensure the columns match the number of values
df_predict = pd.DataFrame([selected_values], columns=['brand', 'model', 'condition', 'warranty', 'screen_size',
                                                      'processor', 'ram', 'card', 'hard_driver', 'made_in', 'hard_driver_kind'])

# Convert 'condition' to categorical before applying the transformation
# df_predict['condition'] = df_predict['condition'].astype('category')

# Now apply the transformer and prediction
deci_predict_laptop = preprocessor.transform(df_predict)

# Make prediction when the user clicks the "Predict Price" button
if st.button("Predict Price"):
    prediction = model.predict(deci_predict_laptop).tolist()
    st.write(f"Predicted Price: {prediction[0]} VND")
