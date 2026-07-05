# K-Means clustering - segmentasi & prioritas penjualan inventoty berdasarkan perputaran stok dan lama penyimpanan stok
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD DATA
print("1. LOAD DATA")

sheet_id = "1__cs0dIpOy9rUekdLE7Dm66fIisDAL6DY1POfneEL1w"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

df = pd.read_csv(url)

# bersihin nama kolom dari spasi tidak kelihatan
df.columns = df.columns.str.strip()

print(f"Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
print("\nDaftar Kolom:")
print(df.columns.tolist())
print("-" * 50)

# 2. PREPROCESSING
print("2. PREPROCESSING")

# 2a. konversi kolom aging dari teks ke angka buat nyari nilai tengah
aging_map = {
    '361-540 (Days)': 450,
    '541-720 (Days)': 630,
    '720+ Prior date': 720
}
df['Aging_Numeric'] = df['Aging'].map(aging_map)
print(f"Konversi Aging ke numerik:")
for k, v in aging_map.items():
    count = (df['Aging'] == k).sum()
    print(f"   '{k}' → {v} hari  ({count} barang)")

# 2b. pilih fitur untuk K-Means
features = ['Inventory: Quantity', 'Inventory: Amount', 'Average unit cost', 'Aging_Numeric']

# PROSES PEMBERSIHAN STRING KE NUMERIK
for col in features:
    if df[col].dtype == 'object':
        # 1. Hapus tanda titik (.) yang merupakan pemisah ribuan
        df[col] = df[col].astype(str).str.replace('.', '', regex=False)
        # 2. Ganti tanda koma (,) menjadi titik (.) untuk standar desimal Python
        df[col] = df[col].str.replace(',', '.', regex=False)
    
    # paksa konversi ke numerik di dataframe uatama
    df[col] = pd.to_numeric(df[col], errors='coerce')

# isi missing values
missing_count_before = df[features].isnull().sum().sum()
if missing_count_before > 0:
    df[features] = df[features].fillna(0)
    print(f"\nCatatan: Ditemukan {missing_count_before} nilai kosong/invalid, otomatis diisi dengan 0.")

X = df[features].copy()

# 2d. normalisasi dengan standard scaler
# split data 80% training, 20% testing
X_train, X_test, idx_train, idx_test = train_test_split(
    X, X.index, test_size=0.2, random_state=42
)

# normalisasi (INGET INI HANYA FIT BUAT SI TRAINING, AWAS PLENGER)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # transform aja, engga di fit ulang

print(f"Data training : {len(X_train)} baris ({len(X_train)/len(X)*100:.0f}%)")
print(f"Data testing  : {len(X_test)} baris ({len(X_test)/len(X)*100:.0f}%)")

# 3. NWNTUIN JUMLAH CLUSTER OPTIMAL (pake ELBOW ama SILHOUETTE)
print("3. MENENTUKAN JUMLAH CLUSTER OPTIMAL")

K_range = range(2, 9)
inertias = []
silhouettes = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_train_scaled)   # ← pakai X_train_scaled
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_train_scaled, labels))

print(f"{'K':>4} | {'Inertia':>12} | {'Silhouette Score':>16}")
print("-" * 40)
for k, i, s in zip(K_range, inertias, silhouettes):
    print(f"{k:>4} | {i:>12.1f} | {s:>16.4f}")

# plot elboy dan silhouette buat grafik
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Penentuan Jumlah Cluster Optimal', fontsize=14, fontweight='bold')

axes[0].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=3, color='red', linestyle='--', label='K=3 (dipilih)')
axes[0].set_xlabel('Jumlah Cluster (K)')
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)')
axes[0].set_title('Elbow Method')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_range), silhouettes, 'gs-', linewidth=2, markersize=8)
axes[1].axvline(x=3, color='red', linestyle='--', label='K=3 (dipilih)')
axes[1].set_xlabel('Jumlah Cluster (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score per K')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
print("\n[Menampilkan grafik penentuan cluster... Tutup jendela grafik untuk melanjutkan ke training model]")
plt.show()
print("-" * 50)

# 4. TRAINING K-MEANS (K=3)
print("4. TRAINING K-MEANS (K=3)")

K_OPTIMAL = 3
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df.loc[idx_train, 'Cluster'] = kmeans.fit_predict(X_train_scaled)   # fit dari training
df.loc[idx_test, 'Cluster']  = kmeans.predict(X_test_scaled)         # predict untuk testing
df['Cluster'] = df['Cluster'].astype(int)

# eval di data testing
sil_train = silhouette_score(X_train_scaled, kmeans.labels_)
sil_test  = silhouette_score(X_test_scaled, kmeans.predict(X_test_scaled))
print(f"Silhouette Score - Training : {sil_train:.4f}")
print(f"Silhouette Score - Testing  : {sil_test:.4f}")
print(f"Inertia (K=3): {kmeans.inertia_:.2f}")

# 5. INTERPRETASI CLUSTER & LABELING PRIORITAS
print("5. INTERPRETASI CLUSTER")

cluster_profile = df.groupby('Cluster')[features].mean()
cluster_profile['Jumlah Barang'] = df.groupby('Cluster').size()
print(cluster_profile.round(2).to_string())

# nentuin label berdasarkan rata-rata Aging_Numeric dan Inventory: Amount per cluster
aging_means = df.groupby('Cluster')['Aging_Numeric'].mean()
amount_means = df.groupby('Cluster')['Inventory: Amount'].mean()

# buat skor prioritas: gabungan aging (bobot 60%) + amount (bobot 40%)
aging_normalized = (aging_means - aging_means.min()) / (aging_means.max() - aging_means.min() + 1e-9)
amount_normalized = (amount_means - amount_means.min()) / (amount_means.max() - amount_means.min() + 1e-9)
priority_score = 0.6 * aging_normalized + 0.4 * amount_normalized

# mastiin engga ada nilai NaN pada priority score sebelum di-ranking
priority_score = priority_score.fillna(0)
priority_rank = priority_score.rank(ascending=False).astype(int)

# mapping nama label berdasarkan rank prioritas (Rank 1 = JUAL SEGERA, Rank 2= PERLU DIPERHATIKSN, Rank 3 = AMAN)
label_map_name = {}
priority_label_map = {}

for cluster_id, rank in priority_rank.items():
    if rank == 1:
        label_map_name[cluster_id] = "JUAL SEGERA"
        priority_label_map[cluster_id] = 1
    elif rank == 2:
        label_map_name[cluster_id] = "PERLU DIPERHATIKAN"
        priority_label_map[cluster_id] = 2
    else:
        label_map_name[cluster_id] = "AMAN"
        priority_label_map[cluster_id] = 3

df['Label'] = df['Cluster'].map(label_map_name)
df['Prioritas'] = df['Cluster'].map(priority_label_map)

print("\nLabel Hasil Otomatisasi per Cluster:")
for c in sorted(df['Cluster'].unique()):
    label = label_map_name[c]
    count = (df['Cluster'] == c).sum()
    avg_aging = df[df['Cluster'] == c]['Aging_Numeric'].mean()
    avg_amount = df[df['Cluster'] == c]['Inventory: Amount'].mean()
    print(f"  Cluster {c} → {label} | {count} barang | Aging rata2: {avg_aging:.0f} hari | Amount rata2: Rp {avg_amount:,.2f}")

print("-" * 50)

# 6. SAVE MODEL
print("6. SAVE MODEL")

pipeline = {
    'kmeans': kmeans,
    'scaler': scaler,
    'aging_map': aging_map,
    'label_map': label_map_name,
    'priority_map': priority_label_map
}
joblib.dump(pipeline, 'model.pkl')