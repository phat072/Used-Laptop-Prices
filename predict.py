import pandas as pd
import streamlit as st
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Định nghĩa các trình biến đổi số và danh mục
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Tạo danh sách các đặc trưng số và danh mục
num_feature = ["warranty", "screen_size", "ram", "hard_driver"]
cat_feature = ["brand", "model", "condition", "processor", "card", "made_in","hard_driver_kind"]

base_dir = os.getcwd()
json_path = os.path.abspath(os.path.join(base_dir, "data/laptop_data_cleaned.csv"))
df = pd.read_csv(json_path)

# Chuyển đổi đơn vị giá từ VND sang triệu VND
df['price_million'] = df['price'] / 1000000  # Thêm cột giá theo triệu VND

# ### Thêm các biểu đồ trực quan hóa dữ liệu ###
st.header("Thống Kê Dữ Liệu")

# Biểu đồ phân phối giá laptop
df_filtered = df[df['price_million'] < 100]
st.subheader("Phân Phối Giá Laptop (Dưới 100 Triệu VND)")
fig_filtered, ax_filtered = plt.subplots()
sns.histplot(df_filtered['price_million'], kde=True, ax=ax_filtered, bins=15, color='orange')
ax_filtered.set_xlabel("Giá (Triệu VND)")
ax_filtered.set_ylabel("Số lượng")
ax_filtered.set_title("Phân Phối Giá Laptop (Dưới 100 Triệu VND)")
st.pyplot(fig_filtered)


# Biểu đồ số lượng sản phẩm theo thương hiệu
st.subheader("Số Lượng Sản Phẩm Theo Thương Hiệu")
brand_counts = df['brand'].value_counts()
fig_brand, ax_brand = plt.subplots(figsize=(10,6))
sns.barplot(x=brand_counts.index, y=brand_counts.values, ax=ax_brand)
ax_brand.set_xlabel("Thương Hiệu")
ax_brand.set_ylabel("Số lượng")
ax_brand.set_title("Số Lượng Sản Phẩm Theo Thương Hiệu")
plt.xticks(rotation=45)
st.pyplot(fig_brand)

# Biểu đồ tình trạng sản phẩm
st.subheader("Tình Trạng Sản Phẩm")
condition_counts = df['condition'].value_counts()
fig_condition, ax_condition = plt.subplots()
sns.barplot(x=condition_counts.index, y=condition_counts.values, ax=ax_condition)
ax_condition.set_xlabel("Tình Trạng")
ax_condition.set_ylabel("Số lượng")
ax_condition.set_title("Tình Trạng Sản Phẩm")
plt.xticks(rotation=45)
st.pyplot(fig_condition)

# Biểu đồ phân phối RAM
st.subheader("Phân Phối RAM")
fig_ram, ax_ram = plt.subplots()
sns.countplot(x='ram', data=df, ax=ax_ram)
ax_ram.set_xlabel("RAM (GB)")
ax_ram.set_ylabel("Số lượng")
ax_ram.set_title("Phân Phối RAM")
st.pyplot(fig_ram)

# Biểu đồ phân phối kích thước màn hình
st.subheader("Phân Phối Kích Thước Màn Hình")
fig_screen, ax_screen = plt.subplots()
sns.histplot(df['screen_size'], bins=10, kde=True, ax=ax_screen)
ax_screen.set_xlabel("Kích Thước Màn Hình (inch)")
ax_screen.set_ylabel("Số lượng")
st.pyplot(fig_screen)


# Danh sách các tệp mô hình để người dùng chọn
model_files = ["model_storage/Decision_Tree_Model_Final.joblib", "model_storage/Ridge_model_Final.joblib", "model_storage/Random_forest_model_Final.joblib"]

# Chọn mô hình dự đoán
selected_model_file = st.selectbox("Chọn Mô Hình Dự Đoán", model_files)

# Định nghĩa bộ tiền xử lý cho các biến đổi
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_feature),
    ('cat', cat_transformer, cat_feature)
])

X = df.drop(columns={"price"})
preprocessor.fit(X)

# Tải mô hình khi người dùng chọn
@st.cache_resource
def load_model(model_file):
    model = joblib.load(model_file)
    return model

model = load_model(selected_model_file)

# Phần chọn thuộc tính sản phẩm
st.header("Chọn Thuộc Tính Sản Phẩm để Dự Đoán")

# Chuyển đổi 'condition' thành kiểu danh mục trước khi xử lý
df["condition"] = df["condition"].astype('category')

# Sử dụng Streamlit selectbox để lấy đầu vào từ người dùng
select_brand = st.selectbox("Chọn Thương Hiệu", df["brand"].unique())
select_model = st.selectbox("Chọn Model", df["model"].unique())
select_condition = st.selectbox("Chọn Tình Trạng", df["condition"].unique()) 
# select_condition = None
select_warranty = st.selectbox("Chọn Bảo Hành", df["warranty"].unique())
select_screen_size = st.selectbox("Chọn Kích Thước Màn Hình", df["screen_size"].unique())
select_processor = st.selectbox("Chọn Bộ Xử Lý", df["processor"].unique())
select_ram = st.selectbox("Chọn RAM", df["ram"].unique())
select_card = st.selectbox("Chọn Card", df["card"].unique())
select_hard_driver = st.selectbox("Chọn Ổ Cứng", df["hard_driver"].unique())
select_made_in = st.selectbox("Chọn Xuất Xứ", df["made_in"].unique())
select_hard_driver_kind = st.selectbox("Chọn Loại Ổ Cứng", df["hard_driver_kind"].unique())

# Tạo danh sách các giá trị đã chọn
selected_values = [
    select_brand, select_model, select_condition, select_warranty, 
    select_screen_size, select_processor, select_ram, select_card,
    select_hard_driver, select_made_in, select_hard_driver_kind
]

# Đảm bảo các cột khớp với số lượng giá trị
df_predict = pd.DataFrame([selected_values], columns=['brand', 'model', 'condition', 'warranty', 'screen_size',
                                                      'processor', 'ram', 'card', 'hard_driver', 'made_in', 'hard_driver_kind'])

# Áp dụng bộ biến đổi và dự đoán
deci_predict_laptop = preprocessor.transform(df_predict)

# Thực hiện dự đoán khi người dùng nhấn nút "Dự Đoán Giá"
if st.button("Dự Đoán Giá"):
    prediction = model.predict(deci_predict_laptop).tolist()
    st.write(f"Giá Dự Đoán: {prediction[0]} VND")
