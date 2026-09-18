# Folder ini kosong secara sengaja

Aplikasi (`app.py`) memuat dataset Olist otomatis dari folder ini saat
dijalankan -- tidak ada widget upload di UI.

Isi folder ini dengan **5 file CSV asli** dari Kaggle:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_geolocation_dataset.csv`

## Cara mengisi folder ini

**Opsi A -- otomatis (butuh akun Kaggle + API token):**
```bash
pip install kaggle
python ../download_data.py    # jalankan dari root proyek, bukan dari dalam folder data/
```

**Opsi B -- manual:**
1. Download dataset dari link Kaggle di atas (tombol "Download").
2. Ekstrak file zip.
3. Salin 5 file CSV di atas persis ke folder `data/` ini (nama file jangan diubah).

Setelah file lengkap, jalankan `streamlit run app.py` dari root proyek,
atau klik tombol **"Muat Ulang Data dari Folder"** di sidebar jika
aplikasi sudah terlanjur berjalan.
