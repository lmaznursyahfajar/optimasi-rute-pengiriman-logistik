# Optimasi Rute Distribusi Last-Mile — Deep Q-Network (DQN)

Studi kasus: **Brazilian E-Commerce Public Dataset by Olist**
(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## Tujuan Bisnis

Olist menghubungkan seller kecil-menengah di Brasil dengan pelanggan;
pengiriman dilakukan lewat mitra logistik. Aplikasi ini menjawab
pertanyaan operasional bagi sebuah hub distribusi (gudang seller):

> Untuk seller tertentu, pada periode tertentu, pelanggan tersebar di mana
> saja — dan bagaimana rute pengiriman yang meminimalkan total jarak
> tempuh armada dibanding heuristik sederhana (nearest-neighbor)?

Aplikasi ini **tidak memakai data simulasi/dummy sama sekali**. Semua
titik pengiriman, KPI performa historis (rata-rata waktu kirim, tingkat
ketepatan waktu, nilai transaksi), dan agregasi demand dihitung langsung
dari transaksi riil pada dataset yang ditempatkan pengguna di folder data/.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi memuat dataset **otomatis dari folder `data/`** di root proyek --
tidak ada widget upload di UI. Jika folder `data/` masih kosong, sidebar
akan menampilkan instruksi kolom apa yang belum lengkap.

## Menyiapkan dataset (sekali saja)

**Opsi A — otomatis lewat Kaggle API (disarankan):**
```bash
pip install kaggle
# siapkan ~/.kaggle/kaggle.json (lihat docstring di download_data.py)
python download_data.py
```

**Opsi B — manual:**
1. Download dataset dari https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Ekstrak, lalu salin 5 file berikut ke folder `data/` (nama file jangan diubah):

| File | Kegunaan di aplikasi |
|---|---|
| `olist_orders_dataset.csv` | status, tanggal pembelian & pengiriman aktual/estimasi |
| `olist_order_items_dataset.csv` | seller per order, nilai produk & ongkos kirim |
| `olist_customers_dataset.csv` | kode pos & kota tujuan pelanggan |
| `olist_sellers_dataset.csv` | kode pos & kota seller (calon hub distribusi) |
| `olist_geolocation_dataset.csv` | **koordinat lat/lng asli per kode pos** (kunci: tidak perlu geocoding online) |

Setelah dataset siap, klik **"Muat Ulang Data dari Folder"** di sidebar
jika aplikasi sudah terlanjur berjalan.

## Struktur

```
app.py                       # UI Streamlit (header tujuan bisnis, sidebar, 3 tab)
download_data.py             # Skrip sekali-jalan: unduh dataset Kaggle -> folder data/
data/                        # Taruh 5 CSV Olist di sini (lihat data/README.md)
modules/
  olist_data.py               # Parsing 5 tabel Olist, KPI bisnis riil, agregasi
                               # demand per kode-pos, koordinat dari geolocation Olist
  data_loader.py               # Matriks jarak & polyline rute riil via OSRM
  dqn_agent.py                 # DQN asli (PyTorch): QNetwork, ReplayBuffer, training loop
  simulator.py                 # Klasterisasi kendaraan, metrik, live tracking
  utils.py                     # Peta Folium: polyline rute riil, warna per kendaraan
requirements.txt
```

## Alur Aplikasi

1. **Dataset dimuat otomatis** dari folder `data/` saat aplikasi start (lihat
   "Menyiapkan dataset" di atas jika belum diisi) — auto-terdeteksi dari nama
   file, tervalidasi kolom minimum tiap tabel, dan di-cache di sesi (tidak
   diproses ulang selama file di disk tidak berubah).
2. **Pilih negara bagian & seller** (hub distribusi), diurutkan berdasarkan volume
   order terkirim.
3. **Pilih rentang tanggal pembelian** — tab *Ringkasan Bisnis* langsung menampilkan
   KPI riil: total order, rata-rata waktu kirim, tingkat ketepatan waktu, total nilai
   produk & ongkos kirim.
4. **Atur jumlah kendaraan & top-N tujuan pelanggan**, lalu klik **Bangun Titik &
   Latih Model DQN** — sistem menyusun titik pengiriman dari koordinat geolocation
   Olist, mengambil matriks jarak riil dari OSRM, melatih DQN, dan menyusun rute
   per kendaraan.
5. Tab **Rute & Pelacakan Live** menampilkan peta dengan polyline rute riil (warna
   per kendaraan), metrik efisiensi vs baseline nearest-neighbor, dan tombol
   **Jalankan Pelacakan Rute (Live)** untuk animasi posisi kendaraan berjalan
   sepanjang rute.

## Keputusan desain penting

1. **Koordinat dari data Olist sendiri, bukan geocoding online.** File
   `olist_geolocation_dataset.csv` sudah berisi lat/lng riil per kode pos —
   jauh lebih andal dan cepat dibanding memanggil API geocoding eksternal,
   dan tidak bergantung pada layanan pihak ketiga untuk data inti.

2. **Hanya order dengan SATU seller yang dipakai.** Satu order Olist bisa
   berisi barang dari beberapa seller berbeda; kasus itu dikecualikan agar
   setiap rute punya SATU hub distribusi yang jelas (lihat docstring
   `build_master_orders` di `olist_data.py`).

3. **Klasterisasi dulu, baru DQN menentukan urutan.** Pembagian pelanggan ke
   kendaraan memakai K-Means geografis (`simulator.py`); DQN menentukan
   urutan optimal *di dalam* satu grup kendaraan (mirip TSP). Full VRP-DQN
   gabungan (klasterisasi + urutan sekaligus dalam satu agent) jauh lebih
   kompleks dan di luar skala wajar untuk aplikasi ini — trade-off yang
   disengaja, bukan keterbatasan yang tidak disadari.

4. **Kapasitas jaringan DQN menyesuaikan otomatis.** `max_group_size`
   dinaikkan otomatis oleh `app.py` jika grup kendaraan terbesar hasil
   klasterisasi melebihi nilai slider sidebar, mencegah crash saat top-N
   pelanggan besar dikombinasikan dengan jumlah kendaraan sedikit.

5. **Baseline nearest-neighbor** dipakai untuk menghitung metrik "efisiensi
   vs baseline" (delta pada `st.metric`), supaya angka yang ditampilkan
   punya makna pembanding, bukan sekadar dekorasi.

6. **Progres training di UI mengikuti proses aktual.** Grafik reward/loss
   dan progress bar saat training dibangun dari `TrainingHistory` yang
   diisi lewat `progress_callback` di setiap episode nyata.

7. **OSRM demo server** (`router.project-osrm.org`) dipakai untuk jarak &
   geometri rute riil (cakupan global termasuk Brasil), gratis tanpa API
   key, tapi rate-limited. Untuk produksi, ganti `OSRM_BASE_URL` di
   `data_loader.py` dengan instance OSRM self-hosted. Jika OSRM tidak
   terjangkau, aplikasi otomatis fallback ke jarak haversine (garis lurus)
   via `geopy` — status sumber data ditampilkan transparan di UI.

## Lisensi Data

Dataset Olist dirilis dengan lisensi CC BY-NC-SA 4.0 oleh Olist di Kaggle.
Aplikasi ini tidak mendistribusikan ulang data tersebut — pengguna
mengunduh dan mengunggahnya sendiri.
