import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression

# Membaca data
data = pd.read_csv('covid_19_indonesia_time_series_all.csv')

# Menyiapkan data untuk analisis densitas penduduk
# Menghapus simbol persen dan mengonversi ke float
data['Case Fatality Rate'] = data['Case Fatality Rate'].str.replace('%', '').astype(float)
data['Case Recovered Rate'] = data['Case Recovered Rate'].str.replace('%', '').astype(float)

# Menyiapkan DataFrame untuk analisis
df = data[['Population Density', 'Case Fatality Rate', 'Case Recovered Rate']].dropna()

# Melatih model regresi untuk tingkat kematian
X_fatality = df[['Population Density']]
y_fatality = df['Case Fatality Rate']
model_fatality = LinearRegression()
model_fatality.fit(X_fatality, y_fatality)
y_pred_fatality = model_fatality.predict(X_fatality)

# Melatih model regresi untuk tingkat pemulihan
X_recovery = df[['Population Density']]
y_recovery = df['Case Recovered Rate']
model_recovery = LinearRegression()
model_recovery.fit(X_recovery, y_recovery)
y_pred_recovery = model_recovery.predict(X_recovery)

# Judul dan deskripsi aplikasi
st.title("Analisis Data COVID-19 di Indonesia")
st.markdown("""
### Pertanyaan Penelitian
1. Bagaimana perbedaan angka kasus aktif antara pulau?
2. Bagaimana densitas penduduk memengaruhi tingkat kematian atau pemulihan?
""")

# Gaya tema Seaborn
sns.set_theme(style="whitegrid", palette="muted")

# Grafik 1: Total Kasus Aktif berdasarkan Pulau
with st.container():
    st.subheader("Total Kasus Aktif Berdasarkan Pulau")
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x='Island', 
        y='Total Active Cases', 
        data=data, 
        palette="viridis"
    )
    plt.title('Total Kasus Aktif Berdasarkan Pulau', fontsize=16, fontweight='bold')
    plt.xlabel('Pulau', fontsize=12)
    plt.ylabel('Total Kasus Aktif', fontsize=12)
    st.pyplot(plt)

# Grafik 2: Pengaruh Densitas Penduduk
with st.container():
    st.subheader("Pengaruh Densitas Penduduk terhadap Tingkat Kematian dan Pemulihan")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    # Scatterplot dan regresi untuk tingkat kematian
    sns.scatterplot(
        ax=axes[0], 
        x='Population Density', 
        y='Case Fatality Rate', 
        data=df, 
        color='blue', 
        label='Data Asli'
    )
    axes[0].plot(df['Population Density'], y_pred_fatality, color='red', label='Garis Regresi')
    axes[0].set_title('Densitas Penduduk vs Tingkat Kematian', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Densitas Penduduk', fontsize=12)
    axes[0].set_ylabel('Tingkat Kematian (%)', fontsize=12)
    axes[0].legend()

    # Scatterplot dan regresi untuk tingkat pemulihan
    sns.scatterplot(
        ax=axes[1], 
        x='Population Density', 
        y='Case Recovered Rate', 
        data=df, 
        color='green', 
        label='Data Asli'
    )
    axes[1].plot(df['Population Density'], y_pred_recovery, color='red', label='Garis Regresi')
    axes[1].set_title('Densitas Penduduk vs Tingkat Pemulihan', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Densitas Penduduk', fontsize=12)
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

# Kesimpulan
st.markdown("""
### Kesimpulan
- **Perbedaan antar pulau:** Cluster tertentu menunjukkan perbedaan signifikan dalam jumlah kasus aktif.
- **Pengaruh densitas penduduk:** 
    - Tingkat kematian cenderung meningkat pada area dengan densitas penduduk tinggi.
    - Tingkat pemulihan menunjukkan pola tertentu berdasarkan densitas penduduk.
""")
