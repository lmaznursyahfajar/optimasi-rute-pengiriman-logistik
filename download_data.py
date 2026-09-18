"""
download_data.py
------------------
Skrip bantu SEKALI JALAN untuk mengunduh dataset Olist dari Kaggle
langsung ke folder data/ pada root proyek ini -- setelah ini, app.py
memuat dataset otomatis dari disk tanpa perlu widget upload sama sekali.

Kenapa skrip terpisah, bukan bagian dari app.py?
Mengunduh dataset Kaggle butuh kredensial API pribadi (kaggle.json) dan
hanya perlu dijalankan SEKALI di komputer Anda -- tidak masuk akal jadi
bagian dari alur Streamlit yang dijalankan berulang kali.

Prasyarat
=========
1. Akun Kaggle (gratis) + API token:
   - Buka https://www.kaggle.com/settings -> bagian "API" -> "Create New Token"
   - Ini akan mengunduh file `kaggle.json`. Simpan ke:
       Linux/Mac : ~/.kaggle/kaggle.json
       Windows   : C:\\Users\\<nama_anda>\\.kaggle\\kaggle.json
   - (Linux/Mac) pastikan permission file aman:
       chmod 600 ~/.kaggle/kaggle.json

2. Install package kaggle:
       pip install kaggle

Menjalankan
===========
Dari root proyek ini (folder yang berisi app.py):
    python download_data.py

Hasil: 5 file CSV Olist akan berada di folder ./data/
"""

from __future__ import annotations

import os
import sys

DATASET_SLUG = "olistbr/brazilian-ecommerce"
DATA_DIR = "data"

REQUIRED_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
]


def main() -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(
            "Package 'kaggle' belum terpasang.\n"
            "Jalankan dulu: pip install kaggle\n"
            "Lalu pastikan kaggle.json sudah ada di ~/.kaggle/ (lihat docstring file ini)."
        )
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        api = KaggleApi()
        api.authenticate()  # membaca ~/.kaggle/kaggle.json
    except Exception as e:
        print(
            f"Gagal autentikasi ke Kaggle: {e}\n"
            "Pastikan file kaggle.json sudah ada di lokasi yang benar (lihat docstring file ini)."
        )
        sys.exit(1)

    print(f"Mengunduh dataset '{DATASET_SLUG}' ke folder '{DATA_DIR}/' ...")
    try:
        api.dataset_download_files(DATASET_SLUG, path=DATA_DIR, unzip=True, quiet=False)
    except Exception as e:
        print(f"Gagal mengunduh dataset: {e}")
        sys.exit(1)

    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"\nPERINGATAN: file berikut tidak ditemukan setelah unduh selesai: {missing}")
        print("Cek isi folder 'data/' secara manual -- mungkin nama file di Kaggle sudah berubah.")
        sys.exit(1)

    print(f"\nSelesai. 5 file Olist siap di folder '{DATA_DIR}/'.")
    print("Sekarang jalankan: streamlit run app.py")


if __name__ == "__main__":
    main()
