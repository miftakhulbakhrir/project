import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import io

# --- 1. Konfigurasi Streamlit ---
st.set_page_config(
    page_title="Aplikasi Prediksi Species Tanaman Iris",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- 2. Fungsi Pemuatan Data dan Pelatihan Model ---

@st.cache_data
def load_data(file_path):
    """Memuat data dan mengembalikan DataFrame."""
    try:
        # Menangani data dengan koma sebagai pemisah desimal jika ada
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error("File 'Iris.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None
    
    # Dekorator st.cache_resource akan menyimpan model agar tidak dilatih ulang setiap kali interaksi
@st.cache_resource
def train_model(df):
    """Melatih model Random Forest Classifier."""
    features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    X = df[features]
    y = df['Species']

    # Scaling fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pembagian data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Inisialisasi dan pelatihan Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, scaler, accuracy, report

# --- Load Data Awal ---
df = load_data("Iris.csv")
if df is None:
    st.stop() # Hentikan eksekusi jika data tidak ditemukan

# --- Pelatihan Model Awal ---
model, scaler, accuracy, report = train_model(df)
features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

# --- SIDEBAR: Input Prediksi ---

st.sidebar.title("Prediksi Species Bunga Iris")
st.sidebar.header("Masukkan Panjang dan Lebar dari Kelopak Serta Mahkota Bunga:")

# Mendapatkan nilai rata-rata, min, dan max untuk nilai default input
mean_values = df[features].mean()

input_data = {}
for feature in features:
    min_val = df[feature].min()
    max_val = df[feature].max()
    default_val = mean_values[feature]
    
    # Mapping label yang lebih mudah dibaca
    label_map = {
        'SepalLengthCm': 'Panjang Kelopak Bunga [cm]',
        'SepalWidthCm': 'Lebar Kelopak Bunga [cm]',
        'PetalLengthCm': 'Panjang Mahkota Bunga [cm]',
        'PetalWidthCm': 'Lebar Mahkota Bunga [cm]',
        
    }

    # Menyesuaikan input berdasarkan jenis fitur
    if feature in ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']:
        input_data[feature] = st.sidebar.slider(
            label_map[feature], 
            min_val.astype(float), 
            max_val.astype(float), 
            float(f'{default_val:.2f}'), 
            step=0.01,
            format="%.2f"
        )
    else:
        input_data[feature] = st.sidebar.number_input(
            label_map[feature], 
            min_value=float(min_val), 
            max_value=float(max_val), 
            value=float(f'{default_val:.2f}'), 
            step=0.01,
            format="%.2f"
        )

    
  # --- MAIN PAGE CONTENT ---

  
images = [
    "iris-setosa.jpg",
    "iris-versicolor.jpg",
    "iris-virginica.jpg"
]

def get_iris_image(species):
    image_urls = {
    "Iris-setosa": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
    "Iris-versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "Iris-virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
    }
    return image_urls.get(species, None)

        # Tombol Prediksi
if st.sidebar.button("Species Bunga Iris"):
    # 1. Konversi input ke DataFrame
    new_data = pd.DataFrame([input_data])
    
    # 2. Scaling data input
    new_data_scaled = scaler.transform(new_data)
    
    # Prediksi
    prediction = model.predict(new_data_scaled)[0]

    # Ambil gambar
    image_path = get_iris_image(prediction)

    # TAMPILAN TENGAH HALAMAN
    st.markdown("---")
    st.markdown(
        "<h2 style='text-align: center;'> Hasil Prediksi Species Bunga Iris </h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<h1 style='text-align: center; color: #2E8B57;'>{prediction.upper()}</h1>",
        unsafe_allow_html=True
    )
    # Tampilkan gambar di tengah
    if image_path:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image_path, use_container_width=True)

    st.success("Prediksi berhasil berdasarkan ukuran kelopak dan mahkota bunga.")
    
st.title("Aplikasi Prediksi Species Bunga Iris")
st.write("""
Selamat Mengeksplorasi Keanekaragaman Species Bunga Iris.
""")
st.markdown("Aplikasi ini menggunakan model *Machine Learning* **Random Forest** untuk memprediksi **Species Bunga Iris** berdasarkan ukuran kelopak dan mahkota bunga")



tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Analisis Data Eksploratif (EDA)", "Pemodelan & Evaluasi", "About Project"])

# ----------------------------------------------------
# TAB 1: Dataset Overview
# ----------------------------------------------------
with tab1:
    st.header("Ringkasan Dataset")
    
    st.subheader("5 Baris Data Pertama")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informasi Data (Tipe Data)")
        # Menangkap output df.info() ke buffer untuk ditampilkan di Streamlit
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
    with col2:
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe())
        
    st.subheader("Cek Nilai Hilang (Missing Values)")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values.rename('Jumlah Nilai Hilang'))
    st.success("Kesimpulan: Dataset bersih, tidak ada nilai yang hilang.")
  
  # ----------------------------------------------------
# TAB 2: Analisis Data Eksploratif (EDA)
# ----------------------------------------------------
with tab2:
    st.header("Analisis Data Eksploratif (EDA)")
    
    st.subheader("Distribusi Species Tanaman")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='Species', data=df, order=df['Species'].value_counts().index, ax=ax, palette='viridis')
    ax.set_title('Distribusi Jumlah Sampel per Species Tanaman')
    st.pyplot(fig)
    st.info("Dataset ini seimbang (*balanced*), dengan jumlah sampel yang sama untuk setiap jenis tanaman.")

    st.subheader("Korelasi Antar Fitur")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Peta Panas Matriks Korelasi Antar Fitur')
    st.pyplot(fig)
    
    st.subheader("Boxplot Karakter Kelopak dan Mahkota Bunga Iris")
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    for i, col in enumerate(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']):
        sns.boxplot(x='Species', y=col, data=df, ax=axes[i], palette='tab10')
        axes[i].set_title(f'Distribusi {col}', fontsize=14)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=90, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    # ----------------------------------------------------
# TAB 3: Modeling & Evaluation
# ----------------------------------------------------
with tab3:
    st.header("Pemodelan Klasifikasi")
    st.subheader("Model yang Digunakan: Random Forest Classifier")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(label="Akurasi Model pada Test Set", value=f"{accuracy*100:.2f}%", delta="Sangat Tinggi")
        st.info("Akurasi sempurna/sangat tinggi karena species bunga memiliki kebutuhan yang sangat spesifik dan terpisah secara jelas.")
    
    with col2:
        st.subheader("Laporan Klasifikasi")
        # Mengubah laporan dari dictionary ke DataFrame untuk tampilan yang lebih baik
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.iloc[:-3, :], use_container_width=True) # Hanya menampilkan metrik per kelas
        
    st.subheader("Analisis Kepentingan Fitur (Feature Importance)")
    
    # Mendapatkan Feature Importance
    rf_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=rf_importances, y=rf_importances.index, palette='rocket', ax=ax)
    ax.set_title('Kepentingan Fitur (Random Forest)')
    st.pyplot(fig)
    
    st.success("Petal Length (cm) dan Petal Width (cm) terbukti menjadi faktor paling penting dalam proses pengambilan keputusan model.")

    # ----------------------------------------------------
# TAB 4: About Project
# ----------------------------------------------------
with tab4:
    st.header("Tentang Proyek Prediksi Species Bunga Iris")
    st.markdown("""
    Proyek ini merupakan pendukung dalam eksplorasi **Jenis Species Bunga Iris** dengan memanfaatkan data untuk proses prediksi yang tepat.
    
    #### Tujuan
    Memberikan hasil prediksi akurat jenis species bunga iris berdasarkan ukuran dari kelopak dan mahkota bunga iris, karena semua jenis species memiliki kemiripan dalam warna mahkota bunga, sehingga kita akan sulit membedakan berdasarkan warna mahkotanya.
    
    #### Detail Teknis
    * **Dataset:** Data ukuran kelopak dan mahkota bungan dari tiga species bunga iris yang berbeda.
    * **Model:** **Random Forest Classifier** (Algoritma *Ensemble* berbasis pohon keputusan).
    * **Metrik Kunci:** Sepal Length (cm), Sepal Width (cm), Petal Length (cm) dan Petal Width (cm).
    
    #### Tentang Pengembang
    Aplikasi ini dikembangkan sebagai studi kasus dalam *Machine Learning* untuk klasifikasi multi-kelas, menunjukkan bagaimana fitur yang memiliki keterpisahan kelas tinggi dapat menghasilkan model prediktif yang hampir sempurna.
    
    ---
    
    *Dibuat dengan Python, Streamlit, dan Scikit-learn.*

    """)
