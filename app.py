# app.py
import streamlit as st

pandas
numpy
matplotlib
seaborn
scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("ABC Manufacturing Revenue Forecasting & Analysis")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Overview")
    st.dataframe(df.head())
    st.write("Data Info:")
    st.text(df.info())
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

    # Step 2: Handle missing data
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Step 3: Convert date & create total_value
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["total_value"] = df["quantity"] * df["unit_price"]

    # Step 4: Visualizations
    st.subheader("Defect Flag Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="defect_flag", ax=ax)
    plt.title("Số lượng sản phẩm defect vs non-defect")
    st.pyplot(fig)

    st.subheader("Revenue Over Time")
    fig2, ax2 = plt.subplots()
    df.groupby("order_date")["total_value"].sum().plot(kind="line", ax=ax2)
    plt.title("Doanh thu theo thời gian")
    plt.ylabel("Tổng giá trị đơn hàng")
    st.pyplot(fig2)

    st.subheader("Machine Temperature vs Quantity")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x="machine_temp_C", y="quantity", hue="defect_flag", ax=ax3)
    plt.title("Ảnh hưởng của nhiệt độ máy đến số lượng sản phẩm & lỗi")
    st.pyplot(fig3)

    st.subheader("Unit Price Distribution by Defect Flag")
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=df, x="defect_flag", y="unit_price", ax=ax4)
    plt.title("So sánh giá đơn vị giữa sản phẩm lỗi và không lỗi")
    st.pyplot(fig4)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    fig5, ax5 = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax5)
    plt.title("Ma trận tương quan giữa các biến số")
    st.pyplot(fig5)

    # Step 5: Prepare features for modeling
    df['ProductionVolume'] = df['quantity'] * np.random.randint(5, 15, size=len(df))
    df['InventoryLevel'] = np.random.randint(50, 500, size=len(df))
    df['SalesOrders'] = np.random.randint(10, 100, size=len(df))
    df['Revenue'] = df['unit_price'] * df['quantity']

    X = df[['ProductionVolume', 'InventoryLevel', 'SalesOrders']]
    y = df['Revenue']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Step 6: Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R²: {r2:.2f}")

    st.success("Analysis and model training completed successfully!")
