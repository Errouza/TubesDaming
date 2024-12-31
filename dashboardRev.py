import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
data = pd.read_csv('covid_19_indonesia_time_series_all.csv')

# Data preprocessing
data['Case Fatality Rate'] = data['Case Fatality Rate'].str.rstrip('%').astype('float') / 100.0
data['Case Recovered Rate'] = data['Case Recovered Rate'].str.rstrip('%').astype('float') / 100.0

# Encoding categorical columns
label_encoder = LabelEncoder()
data['Island'] = label_encoder.fit_transform(data['Island'])
data['Province'] = label_encoder.fit_transform(data['Province'])

# Scaling numerical columns
scaler = StandardScaler()
columns_to_scale = ['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate', 'Case Recovered Rate']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# Sidebar for user input
st.sidebar.title("Dashboard: Supervised & Unsupervised Learning")
option = st.sidebar.selectbox("Select Analysis Type", ("Unsupervised Learning", "Supervised Learning"))

if option == "Unsupervised Learning":
    st.title("Unsupervised Learning: K-Means Clustering")
    
    # K-Means Clustering
    num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[columns_to_scale])
    
    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['Total Cases'], y=data['Total Deaths'], hue=data['Cluster'], palette='viridis')
    plt.title("K-Means Clustering")
    plt.xlabel('Total Cases')
    plt.ylabel('Total Deaths')
    st.pyplot(plt)
    
    st.write("Cluster Centers:")
    st.write(kmeans.cluster_centers_)

    # Insights
    st.write("Insights:")
    st.markdown("""
### Unsupervised Learning: K-Means Clustering
**1. Cluster Analysis:**

Data COVID-19 dikelompokkan ke dalam beberapa cluster berdasarkan fitur seperti Total Cases, Total Deaths, Total Recovered, Population Density, Case Fatality Rate, dan Case Recovered Rate.
Scatter plot menunjukkan distribusi data dalam cluster yang berbeda.
                
**2. Interpretasi Cluster:**

Cluster dengan nilai rata-rata Total Cases dan Total Deaths yang tinggi menunjukkan daerah dengan tingkat infeksi dan kematian yang tinggi.
Cluster dengan tingkat pemulihan yang tinggi menunjukkan daerah dengan respons kesehatan yang lebih baik""")

elif option == "Supervised Learning":
    st.title("Supervised Learning: Random Forest Regressor")
    
    # Handle missing values
    data = data.dropna(subset=['Total Cases'])
    
    # Train-test split
    X = data[columns_to_scale]
    y = data['Total Cases']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_rf_pred = rf_model.predict(X_test)
    
    # Metrics
    rf_mse = mean_squared_error(y_test, y_rf_pred)
    rf_r2 = r2_score(y_test, y_rf_pred)
    st.write("Mean Squared Error:", rf_mse)
    st.write("R2 Score:", rf_r2)
    
    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_rf_pred, alpha=0.5)
    plt.xlabel('Actual Total Cases')
    plt.ylabel('Predicted Total Cases')
    plt.title('Actual vs Predicted Total Cases (Random Forest)')
    st.pyplot(plt)
    
    # Residual plot
    rf_residuals = y_test - y_rf_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_rf_pred, rf_residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Total Cases')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Total Cases (Random Forest)')
    st.pyplot(plt)

    st.write("Insights:")
    st.markdown("""
## Supervised Learning: Random Forest Regressor
**1. Model Performance:**

Mean Squared Error (MSE) rendah dan R2 Score mendekati 1 menunjukkan model prediksi yang baik.

**2. Actual vs Predicted Plot:**

Plot ini menunjukkan hubungan antara nilai aktual dan prediksi Total Cases. Titik yang dekat dengan garis diagonal menunjukkan prediksi yang akurat.

**3.Residual Plot:**

Residual plot menunjukkan kesalahan prediksi. Residual yang tersebar secara acak di sekitar garis horizontal menunjukkan model yang tidak bias.""")