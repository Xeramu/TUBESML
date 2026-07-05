import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import joblib
import io

# CONFIG HALAMAN
st.set_page_config(
    page_title="Segmentasi Inventory - K-Means",
    layout="wide"
)

# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

try:
    pipeline = load_model()
    kmeans      = pipeline['kmeans']
    scaler      = pipeline['scaler']
    aging_map   = pipeline['aging_map']
    label_map   = pipeline['label_map']
    priority_map = pipeline['priority_map']
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# HEADER
st.title("Segmentasi & Prioritas Penjualan Inventory")
st.markdown("**K-Means Clustering** berdasarkan Perputaran Stok dan Lama Penyimpanan Barang")
st.divider()

if not model_loaded:
    st.error("File `inventory_model.pkl` tidak ditemukan! Jalankan dulu `develop_model.py` untuk membuat model.")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.header("Informasi Model")
    st.success("Model berhasil dimuat")
    st.markdown("""
    **Algoritma:** K-Means Clustering  
    **Jumlah Cluster:** 3  
    **Fitur yang digunakan:**
    - Inventory: Quantity
    - Inventory: Amount
    - Average Unit Cost
    - Aging (hari)
    """)
    st.divider()
    st.markdown("""
    **Keterangan Label:**
    - **JUAL SEGERA** → Aging lama, prioritas utama
    - **PERLU DIPERHATIKAN** → Nilai besar, berisiko
    - **AMAN** → Kondisi normal
    """)

# UPLOAD FILE
st.subheader("Upload Data Inventory")
uploaded_file = st.file_uploader(
    "Upload file CSV atau Excel (.xlsx)",
    type=["csv", "xlsx"],
    help="Pastikan file memiliki kolom: Inventory: Quantity, Inventory: Amount, Average unit cost, Aging"
)

if uploaded_file is None:
    st.info("Silakan upload file inventory untuk memulai analisis.")
    st.stop()

# BACA FILE
try:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    st.success(f"File berhasil dimuat: **{df.shape[0]} baris**, **{df.shape[1]} kolom**")
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

# CEK KOLOM WAJIB
required_cols = ['Inventory: Quantity', 'Inventory: Amount', 'Average unit cost', 'Aging']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan: {missing_cols}")
    st.stop()

# PREPROCESSING
# konversi kolom aging ke numerik
df['Aging_Numeric'] = df['Aging'].map(aging_map)
unmapped = df['Aging_Numeric'].isnull().sum()
if unmapped > 0:
    st.warning(f"{unmapped} baris memiliki nilai Aging yang tidak dikenali dan akan diisi dengan median.")
    df['Aging_Numeric'] = df['Aging_Numeric'].fillna(df['Aging_Numeric'].median())

# bersihin kolom numerik
features = ['Inventory: Quantity', 'Inventory: Amount', 'Average unit cost', 'Aging_Numeric']
for col in features:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[features] = df[features].fillna(0)
X = df[features].copy()

# normalisasi & prediksi
X_scaled = scaler.transform(X)
df['Cluster']   = kmeans.predict(X_scaled)
df['Label']     = df['Cluster'].map(label_map)
df['Prioritas'] = df['Cluster'].map(priority_map)

# RINGKASAN HASIL
st.subheader("Ringkasan Hasil Clustering")

col1, col2, col3 = st.columns(3)
jual_segera     = df[df['Label'] == 'JUAL SEGERA']
perlu_perhatian = df[df['Label'] == 'PERLU DIPERHATIKAN']
aman            = df[df['Label'] == 'AMAN']

with col1:
    st.metric("JUAL SEGERA",     f"{len(jual_segera)} barang",
              f"Rp {jual_segera['Inventory: Amount'].sum()/1e9:.1f} Miliar")
with col2:
    st.metric("PERLU DIPERHATIKAN", f"{len(perlu_perhatian)} barang",
              f"Rp {perlu_perhatian['Inventory: Amount'].sum()/1e9:.1f} Miliar")
with col3:
    st.metric("AMAN",            f"{len(aman)} barang",
              f"Rp {aman['Inventory: Amount'].sum()/1e9:.1f} Miliar")

# VISUALISASI
st.subheader("Visualisasi Cluster")

colors = {'JUAL SEGERA': '#e74c3c', 'PERLU DIPERHATIKAN': '#f39c12', 'AMAN': '#2ecc71'}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('K-Means Clustering - Segmentasi Inventory', fontsize=13, fontweight='bold')

# grafik plot 1: scatter agong vs amount
ax1 = axes[0]
for label, color in colors.items():
    subset = df[df['Label'] == label]
    ax1.scatter(subset['Aging_Numeric'], subset['Inventory: Amount'],
                c=color, label=label, alpha=0.6, s=40, edgecolors='none')
ax1.set_xlabel('Aging (Hari)')
ax1.set_ylabel('Inventory Amount (Rp)')
ax1.set_title('Aging vs Inventory Amount')
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'Rp {x/1e6:.0f}jt'))
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# grafik plot 2: jumlah barang per segmen
ax2 = axes[1]
label_counts = df['Label'].value_counts().reindex(['JUAL SEGERA', 'PERLU DIPERHATIKAN', 'AMAN'], fill_value=0)
bar_colors = [colors[l] for l in label_counts.index]
bars = ax2.bar(label_counts.index, label_counts.values, color=bar_colors, edgecolor='white')
ax2.set_title('Jumlah Barang per Segmen')
ax2.set_ylabel('Jumlah Barang')
ax2.set_xticklabels(label_counts.index, fontsize=8)
for bar, val in zip(bars, label_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(val), ha='center', va='bottom', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# grafik plot 3: total amount per segmen
ax3 = axes[2]
amount_per_label = df.groupby('Label')['Inventory: Amount'].sum().reindex(
    ['JUAL SEGERA', 'PERLU DIPERHATIKAN', 'AMAN'], fill_value=0) / 1e9
bar_colors2 = [colors[l] for l in amount_per_label.index]
bars2 = ax3.bar(amount_per_label.index, amount_per_label.values, color=bar_colors2, edgecolor='white')
ax3.set_title('Total Nilai Inventory per Segmen')
ax3.set_ylabel('Total Amount (Miliar Rp)')
ax3.set_xticklabels(amount_per_label.index, fontsize=8)
for bar, val in zip(bars2, amount_per_label.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'Rp {val:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
st.pyplot(fig)
plt.close()

# TABEL PER SEGMEN
st.subheader("Daftar Barang per Segmen")

kolom_tampil = [c for c in ['Product Name', 'Category', 'BU', 'Warehouse',
                             'Inventory: Quantity', 'Inventory: Amount',
                             'Average unit cost', 'Aging', 'PIC'] if c in df.columns]

tab1, tab2, tab3 = st.tabs(["JUAL SEGERA", "PERLU DIPERHATIKAN", "AMAN"])

with tab1:
    data = jual_segera[kolom_tampil].sort_values('Inventory: Amount', ascending=False).reset_index(drop=True)
    st.dataframe(data, use_container_width=True, height=400)
    st.caption(f"Total: {len(data)} barang")

with tab2:
    data = perlu_perhatian[kolom_tampil].sort_values('Inventory: Amount', ascending=False).reset_index(drop=True)
    st.dataframe(data, use_container_width=True, height=400)
    st.caption(f"Total: {len(data)} barang")

with tab3:
    data = aman[kolom_tampil].sort_values('Inventory: Amount', ascending=False).reset_index(drop=True)
    st.dataframe(data, use_container_width=True, height=400)
    st.caption(f"Total: {len(data)} barang")

# DOWNLOAD HASIL
st.subheader("Download Hasil")

output_cols = [c for c in ['Entitas', 'Resource', 'Product Name', 'Category', 'BU',
                            'Site', 'Warehouse', 'Inventory: Quantity', 'Inventory: Amount',
                            'Average unit cost', 'Aging', 'Aging_Numeric', 'PIC',
                            'Cluster', 'Label', 'Prioritas'] if c in df.columns]

df_output = df[output_cols].sort_values(['Prioritas', 'Aging_Numeric', 'Inventory: Amount'],
                                         ascending=[True, False, False])

# excel download
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    df_output.to_excel(writer, sheet_name='Hasil Clustering', index=False)
    df.groupby('Label').agg(
        Jumlah_Barang=('Label', 'count'),
        Total_Amount=('Inventory: Amount', 'sum'),
        Avg_Aging_Hari=('Aging_Numeric', 'mean')
    ).round(2).to_excel(writer, sheet_name='Ringkasan')
    df_output[df_output['Prioritas'] == 1].to_excel(
        writer, sheet_name='Prioritas 1 - Jual Segera', index=False)

st.download_button(
    label="Download Hasil sebagai Excel",
    data=buffer.getvalue(),
    file_name="hasil_clustering_inventory.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)