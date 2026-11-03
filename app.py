
# ============================================================
# APLIKASI WEB MACHINE LEARNING - CLUSTERING
# MALL CUSTOMER SEGMENTATION
#
# PENGEMBANG : HERI SANTOSO
# KELAS      : MTI 08
# NIM        : 26.01.85.7007
# FAKULTAS   : TEKNOLOGI INFORMASI DAN INDUSTRI
# UNIVERSITAS: UNISBANK SEMARANG
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# ============================================================
# HEADER
# ============================================================

st.title("📊 MALL CUSTOMER SEGMENTATION")
st.subheader("Machine Learning - K-Means Clustering")

st.markdown("""
**Pengembang: Heri Santoso**  
MTI 08 | NIM: 26.01.85.7007  
Fakultas Teknologi Informasi dan Industri  
UNISBANK Semarang
""")

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("MENU APLIKASI")

menu = st.sidebar.radio(
    "Pilih Menu:",
    [
        "Dashboard",
        "Data Understanding",
        "EDA",
        "K-Means Clustering",
        "Visualisasi",
        "Download Hasil"
    ]
)

# ============================================================
# DATASET
# ============================================================

st.sidebar.subheader("UPLOAD DATASET")

uploaded_file = st.sidebar.file_uploader(
    "Upload Mall_Customers.csv",
    type=["csv"]
)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = pd.read_csv("Mall_Customers.csv")

except FileNotFoundError:
    st.warning(
        "Silakan upload file Mall_Customers.csv "
        "melalui menu sebelah kiri."
    )
    st.stop()

except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    st.stop()

# ============================================================
# VALIDASI DATASET
# ============================================================

fitur = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

if not all(k in df.columns for k in fitur):
    st.error(
        "Dataset tidak sesuai. Pastikan terdapat kolom "
        "Age, Annual Income (k$), dan Spending Score (1-100)."
    )
    st.stop()

# ============================================================
# PEMBERSIHAN DATA
# ============================================================

data = df.copy()

data = data.drop_duplicates()

for kolom in data.columns:
    if data[kolom].isnull().any():
        if data[kolom].dtype == "object":
            data[kolom] = data[kolom].fillna(
                data[kolom].mode()[0]
            )
        else:
            data[kolom] = data[kolom].fillna(
                data[kolom].median()
            )

# ============================================================
# PROSES CLUSTERING
# ============================================================

st.sidebar.subheader("PENGATURAN CLUSTER")

jumlah_cluster = st.sidebar.slider(
    "Jumlah Cluster (K)",
    min_value=2,
    max_value=10,
    value=5
)

X = data[fitur]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=jumlah_cluster,
    init="k-means++",
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(
    X_scaled,
    data["Cluster"]
)

hasil = data.copy()

# ============================================================
# DASHBOARD
# ============================================================

if menu == "Dashboard":

    st.header("Dashboard Utama")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Pelanggan",
        len(data)
    )

    col2.metric(
        "Jumlah Atribut",
        df.shape[1]
    )

    col3.metric(
        "Jumlah Cluster",
        jumlah_cluster
    )

    st.subheader("Preview Dataset")

    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("Distribusi Jenis Kelamin")

    if "Gender" in data.columns:
        fig, ax = plt.subplots(figsize=(7, 4))

        sns.countplot(
            data=data,
            x="Gender",
            hue="Gender",
            legend=False,
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

    st.success("Dashboard berhasil ditampilkan.")

# ============================================================
# DATA UNDERSTANDING
# ============================================================

elif menu == "Data Understanding":

    st.header("Data Understanding")

    st.write("Nama Dataset: Mall Customer Segmentation")

    st.write("Sumber Dataset: Kaggle")

    st.write("Jumlah Record:", df.shape[0])

    st.write("Jumlah Atribut:", df.shape[1])

    st.subheader("Informasi Dataset")

    info_df = pd.DataFrame({
        "Atribut": df.columns,
        "Tipe Data": df.dtypes.astype(str).values,
        "Jumlah Data": df.notnull().sum().values,
        "Data Kosong": df.isnull().sum().values
    })

    st.dataframe(info_df, use_container_width=True)

    st.subheader("Data 10 Pertama")

    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Data 10 Terakhir")

    st.dataframe(df.tail(10), use_container_width=True)

    st.subheader("Statistik Deskriptif")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

# ============================================================
# EDA
# ============================================================

elif menu == "EDA":

    st.header("Exploratory Data Analysis")

    st.subheader("Statistik Deskriptif")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Pemeriksaan Missing Value")

    st.dataframe(
        df.isnull().sum().to_frame("Jumlah Data Kosong")
    )

    st.subheader("Distribusi Usia")

    fig, ax = plt.subplots()

    sns.histplot(
        data=data,
        x="Age",
        kde=True,
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Distribusi Pendapatan")

    fig, ax = plt.subplots()

    sns.histplot(
        data=data,
        x="Annual Income (k$)",
        kde=True,
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Korelasi Atribut")

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.heatmap(
        data[fitur].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# K-MEANS CLUSTERING
# ============================================================

elif menu == "K-Means Clustering":

    st.header("Implementasi K-Means Clustering")

    st.write(
        "Algoritma K-Means digunakan untuk "
        "mengelompokkan pelanggan berdasarkan usia, "
        "pendapatan, dan spending score."
    )

    st.metric(
        "Silhouette Score",
        round(sil_score, 4)
    )

    st.success(
        f"Clustering berhasil dengan {jumlah_cluster} cluster."
    )

    st.subheader("Hasil Pengelompokan")

    st.dataframe(
        hasil,
        use_container_width=True
    )

    st.subheader("Jumlah Pelanggan Setiap Cluster")

    st.dataframe(
        hasil["Cluster"].value_counts().sort_index()
        .to_frame("Jumlah Pelanggan")
    )

    st.subheader("Profil Rata-rata Cluster")

    st.dataframe(
        hasil.groupby("Cluster")[fitur].mean().round(2),
        use_container_width=True
    )

# ============================================================
# VISUALISASI
# ============================================================

elif menu == "Visualisasi":

    st.header("Visualisasi Hasil Clustering")

    # ELBOW METHOD

    st.subheader("1. Elbow Method")

    wcss = []

    for k in range(1, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)
        wcss.append(model.inertia_)

    fig, ax = plt.subplots()

    ax.plot(
        range(1, 11),
        wcss,
        marker="o",
        color="blue"
    )

    ax.set_xlabel("Jumlah Cluster")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")

    st.pyplot(fig)
    plt.close(fig)

    # SCATTER PLOT

    st.subheader("2. Income vs Spending Score")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=hasil,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="viridis",
        s=100,
        ax=ax
    )

    centroid = scaler.inverse_transform(
        kmeans.cluster_centers_
    )

    ax.scatter(
        centroid[:, 1],
        centroid[:, 2],
        marker="X",
        s=250,
        c="red",
        label="Centroid"
    )

    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    # AGE VS SPENDING

    st.subheader("3. Age vs Spending Score")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=hasil,
        x="Age",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="Set1",
        s=100,
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    # JUMLAH CLUSTER

    st.subheader("4. Jumlah Pelanggan Setiap Cluster")

    jumlah = hasil["Cluster"].value_counts().sort_index()

    fig, ax = plt.subplots()

    jumlah.plot(
        kind="bar",
        color="steelblue",
        ax=ax
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah Pelanggan")

    st.pyplot(fig)
    plt.close(fig)

    # PROFIL CLUSTER

    st.subheader("5. Profil Rata-rata Cluster")

    profil = hasil.groupby("Cluster")[fitur].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    profil.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Nilai Rata-rata")
    ax.set_xlabel("Cluster")

    st.pyplot(fig)
    plt.close(fig)

    # 3D VISUALIZATION

    st.subheader("6. Visualisasi 3D")

    fig = plt.figure(figsize=(10, 7))

    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        hasil["Age"],
        hasil["Annual Income (k$)"],
        hasil["Spending Score (1-100)"],
        c=hasil["Cluster"],
        cmap="viridis",
        s=60
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income")
    ax.set_zlabel("Spending Score")

    fig.colorbar(scatter, ax=ax)

    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# DOWNLOAD HASIL
# ============================================================

elif menu == "Download Hasil":

    st.header("Download Hasil Clustering")

    st.write(
        "Silakan download dataset yang sudah "
        "dilengkapi dengan label cluster."
    )

    csv = hasil.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 DOWNLOAD HASIL CLUSTERING CSV",
        data=csv,
        file_name="Hasil_Clustering_Heri_Santoso.csv",
        mime="text/csv"
    )

    st.success("File hasil clustering siap diunduh.")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="text-align:center;">
<h3>HERI SANTOSO</h3>
<b>MTI 08 | NIM 26.01.85.7007</b><br>
Fakultas Teknologi Informasi dan Industri<br>
UNISBANK SEMARANG<br><br>
<i>Machine Learning - Mall Customer Segmentation</i>
</div>
""", unsafe_allow_html=True)
