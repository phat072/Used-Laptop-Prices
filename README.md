# Product Sales Dashboard from my project about Used Laptop Prices

A Streamlit-based dashboard for visualizing product sales data from a JSON file. The application provides various visualizations to explore and analyze data, such as price distributions, average prices by brand, and feature distributions.

## Features

- **Interactive Upload:** Users can upload a JSON file through the sidebar to populate the dashboard with data.
- **Data Preview:** An expandable section displays the uploaded dataset in a tabular format.
- **Visualizations:**
  - Price distribution by product condition using a box plot.
  - Average price by brand visualized with a bar chart.
  - Screen size distribution represented as a histogram.
  - RAM distribution shown in a pie chart.

## Prerequisites

- Python 3.8 or higher
- Required Python libraries:
  - `streamlit`
  - `pandas`
  - `json`
  - `plotly`

# Product Sales Dashboard and Laptop Price Prediction

This project is a comprehensive solution for analyzing and predicting laptop prices. It includes features for web scraping data, preprocessing, exploratory data analysis (EDA), and building a Streamlit-based interactive dashboard. The data is sourced from [Chợ Tốt](https://www.chotot.com) and processed to provide insights and predictions.

## Features

### 1. Data Collection
- Crawl data from [Chợ Tốt](https://www.chotot.com) using a custom web scraper.
- Extract information like laptop price, brand, screen size, RAM, condition, etc.

### 2. Data Preprocessing
- Clean and preprocess raw data for analysis and modeling.
- Handle missing values, convert data types, and normalize categorical fields.

### 3. Exploratory Data Analysis (EDA)
- Generate visual insights about laptop pricing trends, feature distributions, and correlations.
- Visualizations include:
  - Price distribution by condition and brand.
  - Screen size and RAM distributions.
  - Brand-wise analysis of average prices.

### 4. Price Prediction Model
- Build a machine learning model to predict laptop prices based on features like brand, condition, screen size, RAM, etc.
- Use algorithms such as Linear Regression, Random Forest, or XGBoost.
- Evaluate model performance using metrics like MAE, RMSE, and R².

### 5. Interactive Dashboard
- A user-friendly Streamlit app for data visualization and interaction.
- Includes dynamic charts, a data preview section, and prediction functionality.

## Prerequisites

- Python 3.8 or higher
- Required libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `plotly`
  - `beautifulsoup4`
  - `requests`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/laptop-price-dashboard.git
   cd laptop-price-dashboard
