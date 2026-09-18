"""
app.py
-------
Optimasi Rute Distribusi Last-Mile -- Deep Q-Network (DQN)
Studi kasus: Brazilian E-Commerce Public Dataset by Olist

TUJUAN BISNIS
=============
Olist menghubungkan seller (penjual) kecil-menengah di seluruh Brasil
dengan pelanggan, dan pengiriman dilakukan lewat mitra logistik. Aplikasi
ini menjawab pertanyaan operasional yang nyata bagi sebuah hub distribusi
(gudang seller):

  "Untuk seller tertentu, pada periode tertentu, pelanggan tersebar di
   mana saja -- dan bagaimana rute pengiriman yang meminimalkan total
   jarak tempuh armada dibanding heuristik sederhana (nearest-neighbor)?"

Aplikasi TIDAK menggunakan data simulasi/dummy sama sekali. Seluruh titik,
metrik performa historis (rata-rata waktu kirim, tingkat ketepatan waktu),
dan agregasi demand berasal dari transaksi riil pada dataset Olist yang
disediakan pengguna di folder data/.

Struktur modular:
  modules/olist_data.py  -> parsing 5 tabel Olist, KPI bisnis riil, agregasi
                             demand per kode-pos, koordinat dari geolocation
                             Olist sendiri (TANPA geocoding online)
  modules/data_loader.py -> matriks jarak & polyline rute riil via OSRM
  modules/dqn_agent.py   -> DQN asli (PyTorch): network, replay buffer, training
  modules/simulator.py   -> klasterisasi kendaraan, metrik, live tracking
  modules/utils.py       -> peta Folium (polyline riil, warna per kendaraan)
  app.py (file ini)      -> UI Streamlit, tanpa logika bisnis
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from modules import data_loader as dl
from modules import dqn_agent as dqn
from modules import olist_data as od
from modules import simulator as sim
from modules import utils

st.set_page_config(
    page_title="Optimasi Rute Distribusi Last-Mile -- Olist",
    page_icon="🚚",
    layout="wide",
)


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "master": None,               # tabel gabungan orders+items+customers+sellers
        "geo_lookup": None,           # koordinat median per kode-pos (dari geolocation.csv)
        "points": None,
        "routing_data": None,
        "agent": None,
        "training_history": None,
        "vehicle_groups": None,
        "routes": None,
        "route_geometries": None,
        "sim_states": None,
        "max_group_size": 8,
        "kpi_summary": None,          # BusinessKPISummary terakhir (utk tab Ringkasan Bisnis)
        "skipped_zips": None,
        "selected_seller_label": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()

# Flag tombol aksi -- dideklarasikan di scope modul (bukan trik "in dir()")
# supaya aman direferensikan di bawah terlepas dari cabang sidebar mana yang
# sempat dieksekusi pada run ini.
build_and_train_clicked = False
run_live_clicked = False


# ----------------------------------------------------------------------------
# HEADER -- Tujuan bisnis, ditampilkan jelas di atas
# ----------------------------------------------------------------------------
st.title("🚚 Optimasi Rute Distribusi Last-Mile")
st.caption("Studi kasus: Brazilian E-Commerce Public Dataset by Olist -- Deep Q-Network (PyTorch)")

with st.expander("🎯 Tujuan Bisnis & Cara Kerja Aplikasi", expanded=st.session_state.master is None):
    st.markdown(
        """
        **Masalah bisnis**: sebuah hub distribusi (gudang seller) perlu mengirim pesanan ke
        pelanggan yang tersebar di banyak lokasi. Rute yang tidak efisien memperbesar biaya
        bahan bakar, waktu tempuh, dan risiko keterlambatan.

        **Solusi**: aplikasi ini (1) mengukur performa pengiriman historis seller terpilih
        dari data transaksi riil, lalu (2) melatih agent *Deep Q-Network* untuk menyusun
        rute kunjungan yang meminimalkan total jarak tempuh ke pelanggan tersebut, memakai
        jaringan jalan riil (OSRM) -- dibandingkan langsung dengan baseline heuristik
        *nearest-neighbor* sebagai tolok ukur.

        **Alur penggunaan**: Dataset Olist sudah tersedia otomatis di folder `data/` pada
        root aplikasi (lihat `README.md` jika belum diisi) → pilih negara bagian & seller
        (hub distribusi) di sidebar → pilih rentang tanggal → latih model → lihat rute optimal & efisiensi.
        """
    )


# ----------------------------------------------------------------------------
# SIDEBAR -- Konfigurasi & pemilihan hub distribusi
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Konfigurasi")

    st.subheader("1. Dataset Olist")
    DATA_DIR = "data"
    st.caption(f"Dataset dimuat otomatis dari folder `{DATA_DIR}/` di root aplikasi -- tidak perlu upload manual.")

    reload_clicked = st.button("🔄 Muat Ulang Data dari Folder", use_container_width=True)
    if reload_clicked:
        st.session_state.master = None
        st.session_state.geo_lookup = None
        st.session_state._processed_file_signature = None

    file_map = od.discover_local_files(DATA_DIR)
    missing = set(od.REQUIRED_FILES) - set(file_map)

    if missing:
        st.error(f"File belum lengkap di folder `{DATA_DIR}/`. Tabel yang belum ada: {sorted(missing)}.")
        st.caption("Jalankan `python download_data.py` atau letakkan manual -- lihat README.md.")
        st.session_state.master = None
    else:
        file_signature = od.get_file_signature(file_map)
        if st.session_state.get("_processed_file_signature") != file_signature:
            try:
                with st.spinner("Memuat & menggabungkan tabel Olist (sekali saja, hasil disimpan di sesi)..."):
                    dfs = od.load_olist_files(file_map)
                    master = od.build_master_orders(dfs)
                    geo_lookup = od.build_geolocation_lookup(dfs["geolocation"])
                st.session_state.master = master
                st.session_state.geo_lookup = geo_lookup
                st.session_state._processed_file_signature = file_signature
                st.success(f"{len(master):,} order (single-seller) siap dianalisis.")
            except ValueError as e:
                st.error(f"Gagal memuat data: {e}")
                st.session_state.master = None
        else:
            st.success(f"{len(st.session_state.master):,} order dimuat dari `{DATA_DIR}/`.")

    if st.session_state.master is None:
        st.info(f"Letakkan 5 file CSV Olist di folder `{DATA_DIR}/`, lalu klik **Muat Ulang Data dari Folder**.")
    else:
        master = st.session_state.master

        st.subheader("2. Pilih Hub Distribusi (Seller)")
        states = od.get_states(master)
        sel_state = st.selectbox("Negara Bagian (state) Seller", states)

        seller_ranking = od.get_top_sellers_in_state(master, sel_state, top_n=20)
        if seller_ranking.empty:
            st.warning(f"Tidak ada seller dengan order terkirim di state {sel_state}.")
        else:
            seller_ranking["label"] = seller_ranking.apply(
                lambda r: f"{r['seller_city'].title()} -- {r['seller_id'][:8]}... ({r['n_orders']} order)", axis=1
            )
            sel_label = st.selectbox("Seller (diurutkan berdasarkan volume order)", seller_ranking["label"])
            sel_row = seller_ranking[seller_ranking["label"] == sel_label].iloc[0]
            sel_seller_id = sel_row["seller_id"]
            sel_seller_zip = int(sel_row["seller_zip_code_prefix"])
            st.session_state.selected_seller_label = f"{sel_row['seller_city'].title()} ({sel_seller_id[:8]})"

            st.subheader("3. Rentang Tanggal Pembelian")
            min_date = master["order_purchase_timestamp"].min()
            max_date = master["order_purchase_timestamp"].max()
            date_range = st.date_input("Rentang tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(date_range, tuple) and len(date_range) == 2:
                date_from, date_to = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            else:
                date_from = date_to = pd.Timestamp(date_range)

            filtered, kpi_summary = od.filter_orders_for_seller(master, sel_seller_id, date_from, date_to)
            st.session_state.kpi_summary = kpi_summary

            st.caption(f"{kpi_summary.n_orders:,} order terkirim | {kpi_summary.n_unique_customers_zip} kode pos tujuan")

            if kpi_summary.n_orders == 0:
                st.warning("Tidak ada order pada filter ini -- perlebar rentang tanggal atau pilih seller lain.")
            else:
                st.subheader("4. Armada & Cakupan")
                num_vehicles = st.slider("Jumlah kendaraan/kurir", 1, 5, 3)
                top_n_customers = st.slider("Jumlah tujuan pelanggan (top-N by volume order)", 5, 25, 12)

                with st.expander("Hyperparameter DQN (lanjutan)", expanded=False):
                    episodes = st.slider("Jumlah episode training", 50, 1000, 300, step=50)
                    lr = st.select_slider("Learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
                    gamma = st.slider("Discount factor (gamma)", 0.80, 0.99, 0.95, step=0.01)
                    st.session_state.max_group_size = st.slider(
                        "Kapasitas jaringan minimum", 5, 20, st.session_state.max_group_size
                    )

                build_and_train_clicked = st.button("🧠 Bangun Titik & Latih Model DQN", use_container_width=True, type="primary")
                st.divider()
                run_live_clicked = st.button(
                    "▶️ Jalankan Pelacakan Rute (Live)",
                    use_container_width=True,
                    disabled=st.session_state.agent is None,
                )
                if st.session_state.agent is None:
                    st.caption("Latih model terlebih dahulu sebelum menjalankan pelacakan live.")


# ----------------------------------------------------------------------------
# BUILD + TRAINING (dipicu tombol sidebar). Setiap tab di bawah menangani
# kondisi "belum ada data/model" masing-masing dengan pesan info -- tidak
# ada st.stop() global di sini supaya header & penjelasan tujuan bisnis
# tetap terlihat walau dataset belum lengkap.
# ----------------------------------------------------------------------------
if build_and_train_clicked:
    with st.status("Membangun titik pengiriman & melatih DQN...", expanded=True) as status:
        st.write("Menyusun titik pengiriman dari koordinat geolocation Olist (tanpa geocoding online)...")
        try:
            points, skipped_zips = od.build_points_from_olist(
                filtered, st.session_state.geo_lookup, sel_seller_zip,
                st.session_state.selected_seller_label, top_n_customers,
            )
            st.session_state.points = points
            st.session_state.skipped_zips = skipped_zips
            if skipped_zips:
                st.warning(f"{len(skipped_zips)} kode pos dilewati (tidak ada di tabel geolocation): {skipped_zips}")
        except ValueError as e:
            st.error(str(e))
            status.update(label="❌ Gagal membangun titik pengiriman", state="error")
            st.stop()

        st.write("Mengambil matriks jarak & durasi riil dari OSRM...")
        routing_data = dl.get_distance_matrix(points)
        st.session_state.routing_data = routing_data
        if routing_data.source == "haversine_fallback":
            st.warning("OSRM tidak terjangkau -- menggunakan estimasi jarak garis lurus (haversine) sebagai fallback.")
        else:
            st.write("✅ Matriks jarak riil OSRM berhasil diambil.")

        st.write("Menyusun grup kendaraan (klasterisasi geografis)...")
        vehicle_groups = sim.assign_customers_to_vehicles(points, num_vehicles)
        st.session_state.vehicle_groups = vehicle_groups

        required_group_size = max((len(g) for g in vehicle_groups), default=0) + 1
        effective_max_group_size = max(st.session_state.max_group_size, required_group_size)
        if effective_max_group_size > st.session_state.max_group_size:
            st.info(
                f"Kapasitas jaringan dinaikkan otomatis dari {st.session_state.max_group_size} "
                f"menjadi {effective_max_group_size} agar muat grup kendaraan terbesar."
            )
        st.session_state.max_group_size = effective_max_group_size

        progress_bar = st.progress(0.0)
        metrics_placeholder = st.empty()

        def on_progress(ep, total, reward, loss):
            progress_bar.progress(ep / total)
            if ep % max(1, total // 20) == 0 or ep == total:
                metrics_placeholder.write(f"Episode {ep}/{total} -- reward: {reward:.1f} | loss: {loss:.4f}")

        agent, history = dqn.train_agent(
            distance_matrix=routing_data.distance_matrix,
            max_group_size=effective_max_group_size,
            episodes=episodes,
            lr=lr,
            gamma=gamma,
            progress_callback=on_progress,
        )
        st.session_state.agent = agent
        st.session_state.training_history = history

        st.write("Menyusun rute optimal tiap kendaraan dengan agent terlatih...")
        routes = sim.build_routes_with_agent(agent, routing_data.distance_matrix, 0, vehicle_groups, effective_max_group_size)
        st.session_state.routes = routes

        st.write("Mengambil geometri jalan riil (polyline) tiap rute dari OSRM...")
        route_geoms = [dl.build_full_route_geometry(points, r) for r in routes]
        st.session_state.route_geometries = route_geoms

        st.session_state.sim_states = [
            sim.VehicleSimState(vehicle_id=i, route=routes[i], path_coords=route_geoms[i]) for i in range(len(routes))
        ]

        status.update(label="✅ Model terlatih & rute siap!", state="complete", expanded=False)


# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_bisnis, tab_train, tab_map = st.tabs(["📊 Ringkasan Bisnis", "🧠 Training DQN", "🗺️ Rute & Pelacakan Live"])


# ---- TAB: RINGKASAN BISNIS ----
with tab_bisnis:
    if st.session_state.kpi_summary is None:
        st.info("Lengkapi dataset di folder `data/` dan pilih seller di sidebar untuk melihat ringkasan performa bisnis.")
    else:
        k = st.session_state.kpi_summary
        st.subheader(f"Performa Historis: {st.session_state.selected_seller_label}")
        st.caption(f"Periode: {k.date_min.date()} s/d {k.date_max.date()}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Order Terkirim", f"{k.n_orders:,}")
        c2.metric("Rata-rata Waktu Kirim", f"{k.avg_delivery_days:.1f} hari")
        c3.metric(
            "Tingkat Ketepatan Waktu",
            f"{k.ontime_rate * 100:.1f}%",
            delta="Baik" if k.ontime_rate >= 0.9 else "Perlu perbaikan",
            delta_color="normal" if k.ontime_rate >= 0.9 else "inverse",
        )
        c4.metric("Total Nilai Produk", f"R$ {k.total_product_value:,.0f}")

        st.metric("Total Biaya Ongkos Kirim (freight)", f"R$ {k.total_freight_value:,.0f}")

        if st.session_state.points is not None:
            st.subheader("Sebaran Titik Pengiriman")
            preview_map = utils.build_base_map(st.session_state.points)
            st_folium(preview_map, use_container_width=True, height=420, key="preview_map")
            if st.session_state.skipped_zips:
                st.caption(
                    f"Catatan kualitas data: {len(st.session_state.skipped_zips)} kode pos pelanggan "
                    "dilewati karena tidak ditemukan di tabel geolocation Olist."
                )


# ---- TAB: TRAINING ----
with tab_train:
    st.subheader("Kurva Training DQN")
    history = st.session_state.training_history

    if history is None:
        st.info("Klik **Bangun Titik & Latih Model DQN** di sidebar untuk memulai.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history.episode_rewards, color="#2A9D8F")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.set_title("Reward per Episode (aktual, dari proses training)")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(history.episode_losses, color="#E63946")
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Rata-rata Loss (MSE)")
            ax2.set_title("Loss per Episode (aktual, dari proses training)")
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

        final_reward = history.episode_rewards[-1]
        best_reward = max(history.episode_rewards)
        c1, c2 = st.columns(2)
        c1.metric("Reward Episode Terakhir", f"{final_reward:.1f}")
        c2.metric("Reward Terbaik", f"{best_reward:.1f}", delta=f"{final_reward - history.episode_rewards[0]:.1f} vs awal")


# ---- TAB: PETA & PELACAKAN LIVE ----
with tab_map:
    if st.session_state.routes is None:
        st.info("Latih model untuk melihat rute optimal & menjalankan pelacakan live.")
    else:
        points = st.session_state.points
        routes = st.session_state.routes
        route_geoms = st.session_state.route_geometries
        routing_data = st.session_state.routing_data

        st.markdown(utils.legend_html(len(routes)), unsafe_allow_html=True)

        dqn_total_dist = sum(sim.route_total_distance(routing_data.distance_matrix, r) for r in routes)
        baseline_routes = [
            sim.nearest_neighbor_baseline(routing_data.distance_matrix, 0, g) for g in st.session_state.vehicle_groups
        ]
        baseline_total_dist = sum(sim.route_total_distance(routing_data.distance_matrix, r) for r in baseline_routes)
        improvement_pct = 100 * (baseline_total_dist - dqn_total_dist) / baseline_total_dist if baseline_total_dist > 0 else 0
        dqn_total_time_min = sum(sim.route_total_duration(routing_data.duration_matrix, r) for r in routes) / 60

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Jarak Rute DQN", f"{dqn_total_dist/1000:.1f} km")
        m2.metric(
            "Efisiensi vs Nearest-Neighbor",
            f"{improvement_pct:+.1f}%",
            delta=f"{(baseline_total_dist - dqn_total_dist)/1000:+.1f} km",
            delta_color="normal" if improvement_pct >= 0 else "inverse",
        )
        m3.metric("Estimasi Total Waktu Tempuh", f"{dqn_total_time_min:.0f} menit")

        map_placeholder = st.empty()
        status_placeholder = st.empty()

        def render_static_map():
            fmap = utils.build_base_map(points)
            for i, route in enumerate(routes):
                delivered = st.session_state.sim_states[i].delivered if st.session_state.sim_states else []
                pos = st.session_state.sim_states[i].current_position if st.session_state.sim_states else None
                utils.add_route_to_map(fmap, points, i, route_geoms[i], delivered_nodes=delivered, vehicle_position=pos)
            with map_placeholder:
                st_folium(fmap, use_container_width=True, height=520, key=f"map_{time.time()}")

        def render_status_table():
            rows = []
            for vs in st.session_state.sim_states:
                total_stops = len(vs.route) - 2
                rows.append(
                    {
                        "Kendaraan": f"#{vs.vehicle_id + 1}",
                        "Total Stop": total_stops,
                        "Terkirim": len(vs.delivered),
                        "Status": "✅ Selesai" if vs.finished else "🚚 Dalam perjalanan",
                    }
                )
            with status_placeholder:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        render_static_map()
        render_status_table()

        if run_live_clicked:
            st.session_state.sim_states = [
                sim.VehicleSimState(vehicle_id=i, route=routes[i], path_coords=route_geoms[i]) for i in range(len(routes))
            ]
            max_len = max((len(vs.path_coords) for vs in st.session_state.sim_states), default=1)
            step_size = max(1, max_len // 40)

            progress = st.progress(0.0)
            frame = 0
            total_frames = max_len // step_size + 2

            while not all(vs.finished for vs in st.session_state.sim_states):
                for vs in st.session_state.sim_states:
                    sim.advance_vehicle(vs, points, step_size)
                render_static_map()
                render_status_table()
                frame += 1
                progress.progress(min(1.0, frame / total_frames))
                time.sleep(0.4)

            st.success("Pelacakan selesai -- seluruh kendaraan telah menyelesaikan rutenya.")
