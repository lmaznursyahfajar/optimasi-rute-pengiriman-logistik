"""
modules/olist_data.py
------------------------
Jalur data UTAMA aplikasi: Brazilian E-Commerce Public Dataset by Olist
(Kaggle: olistbr/brazilian-ecommerce). Tidak ada data simulasi/dummy di
modul ini -- semua angka berasal dari transaksi riil yang disediakan
pengguna.

Tujuan bisnis yang dijawab modul ini:
  "Untuk seller (hub distribusi) tertentu, pada periode tertentu, kemana
   saja pelanggan tersebar, apa performa pengiriman historisnya (rata-rata
   waktu tempuh, tingkat ketepatan waktu), dan seperti apa rute distribusi
   yang optimal untuk melayani sebaran pelanggan tersebut?"

Alur:
  1. load_olist_files()        -> parse & validasi 5 CSV wajib
  2. build_master_orders()     -> gabungkan orders+items+customers+sellers,
                                   hanya order dgn SATU seller (untuk depot
                                   yang jelas), hitung metrik ketepatan waktu
  3. get_states() / get_top_sellers_in_state() -> untuk dropdown pemilihan
     hub distribusi
  4. filter_orders_for_seller() -> subset + BusinessKPISummary riil
  5. aggregate_customer_demand() -> jumlah order per kode-pos pelanggan
  6. build_geolocation_lookup() -> koordinat median per kode-pos (SEKALI
     hitung, dari file geolocation yang sama-sama disediakan -- BUKAN dari
     API eksternal)
  7. build_points_from_olist() -> gabungkan semua -> DataFrame `points`
     (id, name, lat, lon, demand, is_depot) siap dipakai dqn_agent.py /
     simulator.py / utils.py TANPA modifikasi apa pun pada modul tsb.

Kenapa hanya order dengan SATU seller?
Satu order bisa berisi barang dari beberapa seller berbeda (multi-seller
order) -- dalam kasus itu "depot" pengirimannya ambigu (dikirim dari mana?
oleh siapa?). Untuk menjaga model rute tetap punya SATU hub yang jelas per
kendaraan, order multi-seller dikecualikan dari agregasi demand. Ini
trade-off yang disengaja: mengorbankan sebagian data demi definisi masalah
routing yang valid, bukan keterbatasan yang tidak disadari.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

REQUIRED_FILES = {
    "orders": {
        "order_id", "customer_id", "order_status", "order_purchase_timestamp",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    },
    "order_items": {"order_id", "product_id", "seller_id", "price", "freight_value"},
    "customers": {"customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"},
    "sellers": {"seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"},
    "geolocation": {"geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"},
}

# Pola nama file untuk auto-deteksi tabel dari folder data/ lokal (urutan
# penting: "order_items" dicek sebelum kunci lain yang lebih pendek/umum).
FILENAME_PATTERNS = [
    ("order_items", "order_items"),
    ("orders", "orders"),
    ("customers", "customers"),
    ("sellers", "sellers"),
    ("geolocation", "geolocation"),
]


@dataclass
class BusinessKPISummary:
    """Ringkasan performa bisnis RIIL (bukan simulasi) untuk seller + filter
    tanggal yang dipilih -- dipakai tab Ringkasan Bisnis sebagai baseline
    sebelum optimasi rute.
    """

    n_orders: int
    n_unique_customers_zip: int
    avg_delivery_days: float
    ontime_rate: float
    total_freight_value: float
    total_product_value: float
    date_min: pd.Timestamp
    date_max: pd.Timestamp


def discover_local_files(data_dir: str) -> dict[str, str]:
    """Cari 5 file Olist di folder lokal pada root aplikasi, memakai pola
    nama file yang sama (lihat FILENAME_PATTERNS di atas). Dipakai karena
    aplikasi ini memuat dataset langsung dari disk -- tidak ada widget
    widget upload sama sekali (dimuat dari folder data/).

    Return: dict {nama_tabel: path_file}. Tabel yang tidak ditemukan tidak
    muncul di dict -- pemanggil (app.py) yang memutuskan pesan apa yang
    ditampilkan ke pengguna untuk tabel yang hilang.
    """
    if not os.path.isdir(data_dir):
        return {}

    detected: dict[str, str] = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        lower = fname.lower()
        for key, pattern in FILENAME_PATTERNS:
            if pattern in lower and key not in detected:
                detected[key] = os.path.join(data_dir, fname)
                break
    return detected


def get_file_signature(file_map: dict[str, str]) -> tuple:
    """Signature ringan (path + waktu modifikasi + ukuran) untuk file lokal,
    dipakai app.py agar tabel besar (geolocation ~1 juta baris) TIDAK
    diproses ulang pada setiap interaksi widget -- hanya saat file di disk
    benar-benar berubah atau pengguna menekan tombol 'Muat Ulang Data'.
    """
    sig = []
    for key in sorted(file_map):
        path = file_map[key]
        try:
            stat = os.stat(path)
            sig.append((key, path, stat.st_mtime, stat.st_size))
        except OSError:
            sig.append((key, path, None, None))
    return tuple(sig)


def load_olist_files(file_map: dict[str, object]) -> dict[str, pd.DataFrame]:
    """Parse 5 CSV wajib, validasi kolom minimum tiap tabel. Melempar
    ValueError dengan pesan jelas (nama tabel + kolom yang hilang) jika
    format tidak sesuai skema resmi Olist.
    """
    missing_tables = set(REQUIRED_FILES) - set(file_map)
    if missing_tables:
        raise ValueError(f"File belum lengkap di folder data/. Tabel yang belum ditemukan: {sorted(missing_tables)}")

    dfs: dict[str, pd.DataFrame] = {}
    for table, required_cols in REQUIRED_FILES.items():
        df = pd.read_csv(file_map[table])
        df.columns = [c.strip() for c in df.columns]
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Tabel '{table}': kolom hilang {sorted(missing_cols)}. Pastikan file Olist asli.")
        dfs[table] = df

    return dfs


def build_master_orders(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Gabungkan orders + order_items (single-seller only) + customers +
    sellers menjadi satu tabel master, plus kolom turunan performa
    pengiriman (delivery_days, on_time) untuk KPI bisnis.
    """
    orders = dfs["orders"].copy()
    items = dfs["order_items"].copy()
    customers = dfs["customers"].copy()
    sellers = dfs["sellers"].copy()

    # Hanya order dengan tepat satu seller unik (lihat penjelasan di docstring modul).
    seller_per_order = items.groupby("order_id")["seller_id"].agg(n_seller="nunique", seller_id="first")
    freight_per_order = items.groupby("order_id")["freight_value"].sum().rename("total_freight_value")
    price_per_order = items.groupby("order_id")["price"].sum().rename("total_product_value")

    single_seller_orders = seller_per_order[seller_per_order["n_seller"] == 1][["seller_id"]].reset_index()

    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
    orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

    master = orders.merge(single_seller_orders, on="order_id", how="inner")
    master = master.merge(freight_per_order, on="order_id", how="left")
    master = master.merge(price_per_order, on="order_id", how="left")
    master = master.merge(
        customers[["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"]],
        on="customer_id", how="left",
    )
    master = master.merge(
        sellers[["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]],
        on="seller_id", how="left",
    )

    master["is_delivered"] = master["order_status"].eq("delivered") & master["order_delivered_customer_date"].notna()
    master["delivery_days"] = (
        master["order_delivered_customer_date"] - master["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    master["on_time"] = master["order_delivered_customer_date"] <= master["order_estimated_delivery_date"]

    return master


def get_states(master: pd.DataFrame) -> list[str]:
    return sorted(master["seller_state"].dropna().unique().tolist())


def get_top_sellers_in_state(master: pd.DataFrame, state: str, top_n: int = 20) -> pd.DataFrame:
    """Ranking seller (calon hub distribusi) berdasarkan volume order
    terkirim di satu state, untuk dropdown pemilihan hub."""
    subset = master[(master["seller_state"] == state) & master["is_delivered"]]
    ranking = (
        subset.groupby(["seller_id", "seller_city", "seller_zip_code_prefix"])
        .size()
        .reset_index(name="n_orders")
        .sort_values("n_orders", ascending=False)
        .head(top_n)
    )
    return ranking


def filter_orders_for_seller(
    master: pd.DataFrame,
    seller_id: str,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
) -> tuple[pd.DataFrame, BusinessKPISummary]:
    """Subset order TERKIRIM (delivered) untuk satu seller pada rentang
    tanggal pembelian tertentu, plus ringkasan KPI bisnis riilnya.
    """
    mask = (
        (master["seller_id"] == seller_id)
        & master["is_delivered"]
        & (master["order_purchase_timestamp"] >= date_from)
        & (master["order_purchase_timestamp"] <= date_to)
    )
    filtered = master.loc[mask].copy()

    summary = BusinessKPISummary(
        n_orders=len(filtered),
        n_unique_customers_zip=filtered["customer_zip_code_prefix"].nunique(),
        avg_delivery_days=float(filtered["delivery_days"].mean()) if len(filtered) else 0.0,
        ontime_rate=float(filtered["on_time"].mean()) if len(filtered) else 0.0,
        total_freight_value=float(filtered["total_freight_value"].sum()) if len(filtered) else 0.0,
        total_product_value=float(filtered["total_product_value"].sum()) if len(filtered) else 0.0,
        date_min=filtered["order_purchase_timestamp"].min() if len(filtered) else date_from,
        date_max=filtered["order_purchase_timestamp"].max() if len(filtered) else date_to,
    )
    return filtered, summary


def aggregate_customer_demand(filtered: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Jumlah order (demand) per kode-pos pelanggan, diambil top-N
    tersibuk -- membatasi ukuran graf routing agar tetap wajar untuk DQN.
    """
    if filtered.empty:
        return pd.DataFrame(columns=["customer_zip_code_prefix", "customer_city", "demand"])

    agg = (
        filtered.groupby("customer_zip_code_prefix")
        .agg(demand=("order_id", "count"), customer_city=("customer_city", "first"))
        .reset_index()
        .sort_values("demand", ascending=False)
    )
    return agg.head(top_n).reset_index(drop=True)


def build_geolocation_lookup(geo_df: pd.DataFrame) -> pd.DataFrame:
    """Koordinat representatif (median, tahan outlier) per kode-pos,
    dihitung SEKALI dari file geolocation yang disediakan pengguna sendiri --
    tidak ada panggilan API eksternal sama sekali di jalur data ini.
    """
    lookup = (
        geo_df.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lon"})
    )
    return lookup


def build_points_from_olist(
    filtered: pd.DataFrame,
    geo_lookup: pd.DataFrame,
    seller_zip_prefix: int,
    seller_label: str,
    top_n: int,
) -> tuple[pd.DataFrame, list[int]]:
    """Rangkai semua langkah menjadi DataFrame `points` dengan skema PERSIS
    SAMA seperti sebelumnya (id, name, lat, lon, demand, is_depot).
    Return juga daftar kode-pos yang terpaksa dilewati karena tidak
    ditemukan di tabel geolocation (transparansi kualitas data).
    """
    agg = aggregate_customer_demand(filtered, top_n=top_n)
    if agg.empty:
        raise ValueError("Tidak ada order pada filter seller/tanggal ini. Perlebar rentang tanggal atau pilih seller lain.")

    if seller_zip_prefix not in geo_lookup.index:
        raise ValueError(
            f"Koordinat hub (kode pos {seller_zip_prefix}) tidak ditemukan di tabel geolocation. "
            "Pilih seller lain."
        )

    depot_lat = float(geo_lookup.loc[seller_zip_prefix, "lat"])
    depot_lon = float(geo_lookup.loc[seller_zip_prefix, "lon"])

    rows = [
        {
            "id": 0,
            "name": f"Hub Distribusi -- {seller_label}",
            "lat": depot_lat,
            "lon": depot_lon,
            "demand": 0,
            "is_depot": True,
        }
    ]

    skipped_zips: list[int] = []
    idx = 1
    for _, r in agg.iterrows():
        zip_prefix = int(r["customer_zip_code_prefix"])
        if zip_prefix not in geo_lookup.index:
            skipped_zips.append(zip_prefix)
            continue
        rows.append(
            {
                "id": idx,
                "name": f"{str(r['customer_city']).title()} ({zip_prefix})",
                "lat": float(geo_lookup.loc[zip_prefix, "lat"]),
                "lon": float(geo_lookup.loc[zip_prefix, "lon"]),
                "demand": int(r["demand"]),
                "is_depot": False,
            }
        )
        idx += 1

    if idx == 1:
        raise ValueError("Semua kode pos pelanggan pada filter ini tidak ditemukan di tabel geolocation.")

    return pd.DataFrame(rows), skipped_zips
