streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("ABC Manufacturing Revenue Forecasting")

# Upload dữ liệu
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Overview")
    st.dataframe(df.head())

    # Xử lý dữ liệu thiếu
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Chuyển đổi dữ liệu
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["total_value"] = df["quantity"] * df["unit_price"]

    # Tạo biến giả lập
    df['ProductionVolume'] = df['quantity'] * np.random.randint(5, 15, size=len(df))
    df['InventoryLevel'] = np.random.randint(50, 500, size=len(df))
    df['SalesOrders'] = np.random.randint(10, 100, size=len(df))
    df['Revenue'] = df['unit_price'] * df['quantity']

    # Xác định features và target
    X = df[['ProductionVolume', 'InventoryLevel', 'SalesOrders']]
    y = df['Revenue']

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Hiển thị kết quả
    st.subheader("Model Evaluation")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    st.write(f"R²: {r2_score(y_test, y_pred):.2f}")

    # Biểu đồ trực quan
    st.subheader("Revenue vs Production Volume")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='ProductionVolume', y='Revenue', ax=ax)
    st.pyplot(fig)
