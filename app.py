from __future__ import annotations

import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from modules import data_loader as dl
from modules import dqn_agent as dqn
from modules import order_data as od
from modules import simulator as sim
from modules import utils

st.set_page_config(page_title="Optimasi Rute Distribusi Dinamis - DQN", page_icon="🚚", layout="wide")


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "points": None,
        "routing_data": None,          # RoutingData (distance/duration matrix)
        "agent": None,
        "training_history": None,
        "vehicle_groups": None,
        "routes": None,                # list[list[int]] indeks global per kendaraan
        "route_geometries": None,      # list[list[[lat,lon]]] polyline riil per kendaraan
        "sim_states": None,            # list[VehicleSimState]
        "sim_running": False,
        "max_group_size": 8,
        "order_df": None,              # DataFrame log order mentah (setelah upload)
        "order_build_info": None,      # dict ringkasan filter+agregasi terakhir (utk tab Data)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")

    st.subheader("1. Sumber Data Titik Pengiriman")
    data_source = st.radio(
        "Pilih sumber data",
        ["Sampel Jakarta", "Koordinat Kustom (lat/lon)", "Data Historis Pesanan (order log)"],
        key="data_source_radio",
    )

    # --- Sumber 1: sampel bawaan ---
    if data_source == "Sampel Jakarta":
        st.session_state.points = dl.load_sample_data()
        st.caption("15 titik sampel Jakarta (depot + 14 toko).")

    # --- Sumber 2: CSV koordinat langsung (perilaku lama, tidak berubah) ---
    elif data_source == "Koordinat Kustom (lat/lon)":
        uploaded_file = st.file_uploader(
            "CSV (kolom: id, name, lat, lon, demand, is_depot)", type=["csv"], key="coord_uploader"
        )
        if uploaded_file is not None:
            try:
                st.session_state.points = dl.load_uploaded_data(uploaded_file)
                st.success(f"{len(st.session_state.points)} titik berhasil dimuat dari CSV Anda.")
            except ValueError as e:
                st.error(f"CSV tidak valid: {e}")
                st.session_state.points = dl.load_sample_data()
        else:
            st.session_state.points = dl.load_sample_data()
            st.caption("Belum ada upload -- menggunakan data sampel Jakarta sementara.")

    # --- Sumber 3: log historis pesanan e-commerce (TANPA koordinat) ---
    else:
        order_file = st.file_uploader(
            "CSV order log (order_id, order_date, city, district, ontime, dst.)",
            type=["csv"],
            key="order_uploader",
        )
        if order_file is not None:
            try:
                st.session_state.order_df = od.load_order_history(order_file)
                st.success(f"{len(st.session_state.order_df):,} baris order dimuat.")
            except ValueError as e:
                st.error(f"CSV tidak valid: {e}")
                st.session_state.order_df = None

        if st.session_state.order_df is None:
            st.session_state.points = dl.load_sample_data()
            st.caption("Belum ada upload order log -- menggunakan data sampel Jakarta sementara.")
        else:
            order_df = st.session_state.order_df
            cities = od.get_available_cities(order_df)
            sel_city = st.selectbox("Kota", cities)

            min_date = order_df["order_date"].min()
            max_date = order_df["order_date"].max()
            date_range = st.date_input(
                "Rentang tanggal order",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                date_from, date_to = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            else:
                date_from = date_to = pd.Timestamp(date_range)

            top_n_districts = st.slider("Jumlah kecamatan teratas (by jumlah order)", 5, 20, 12)

            filtered, summary = od.filter_orders(order_df, sel_city, date_from, date_to)
            st.caption(
                f"{summary.n_orders:,} order | {summary.n_districts} kecamatan | "
                f"tingkat ontime {summary.ontime_rate * 100:.1f}%"
            )

            if summary.n_orders == 0:
                st.warning("Tidak ada order pada filter kota/tanggal ini -- perlebar rentang tanggal.")
            else:
                preview_agg = od.aggregate_district_demand(filtered, top_n=top_n_districts)
                depot_options = ["Otomatis (kecamatan tersibuk)"] + preview_agg["district"].tolist()
                depot_choice = st.selectbox("Depot / gudang di kecamatan", depot_options)
                depot_district = None if depot_choice.startswith("Otomatis") else depot_choice

                build_points_clicked = st.button(
                    "📍 Bangun Titik Pengiriman (Geocoding)", use_container_width=True
                )
                if build_points_clicked:
                    n_to_geocode = len(preview_agg) + (
                        1 if depot_district is not None and depot_district not in preview_agg["district"].values else 0
                    )
                    with st.status(
                        f"Geocoding {n_to_geocode} kecamatan via Nominatim (OpenStreetMap)...",
                        expanded=True,
                    ) as geostat:
                        st.write(
                            "Kebijakan Nominatim membatasi 1 request/detik -- proses ini "
                            "butuh beberapa detik per kecamatan BARU (hasil di-cache di "
                            "`geocode_cache.csv`, sesi berikutnya untuk kecamatan yang sama "
                            "akan instan)."
                        )
                        geo_progress = st.progress(0.0)
                        geo_log = st.empty()

                        def on_geo_progress(i, n, district):
                            geo_progress.progress(i / n)
                            geo_log.write(f"({i}/{n}) {district}")

                        try:
                            new_points = od.build_points_from_orders(
                                filtered, sel_city, top_n_districts, depot_district, on_geo_progress
                            )
                            st.session_state.points = new_points
                            st.session_state.order_build_info = {
                                "city": sel_city,
                                "date_from": date_from,
                                "date_to": date_to,
                                "summary": summary,
                                "agg": preview_agg,
                            }
                            geostat.update(
                                label=f"✅ {len(new_points)} titik pengiriman siap untuk {sel_city}.",
                                state="complete",
                                expanded=False,
                            )
                        except ValueError as e:
                            st.error(str(e))
                            geostat.update(label="❌ Gagal membangun titik pengiriman", state="error")

            if st.session_state.points is None:
                st.session_state.points = dl.load_sample_data()

    st.subheader("2. Armada")
    num_vehicles = st.slider("Jumlah kendaraan/kurir", min_value=1, max_value=5, value=3)

    st.subheader("3. Hyperparameter DQN")
    with st.expander("Atur parameter training", expanded=False):
        episodes = st.slider("Jumlah episode", 50, 1000, 300, step=50)
        lr = st.select_slider("Learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
        gamma = st.slider("Discount factor (gamma)", 0.80, 0.99, 0.95, step=0.01)
        st.session_state.max_group_size = st.slider(
            "Ukuran grup maksimum (kapasitas jaringan)", 5, 15, st.session_state.max_group_size
        )

    train_clicked = st.button("🧠 Latih Model DQN", use_container_width=True)
    st.divider()
    sim_clicked = st.button(
        "▶️ Mulai Simulasi",
        use_container_width=True,
        disabled=st.session_state.agent is None,
        type="primary",
    )
    if st.session_state.agent is None:
        st.caption("Latih model terlebih dahulu sebelum menjalankan simulasi.")


points = st.session_state.points
depot_idx = int(points.index[points["is_depot"]][0])


# ----------------------------------------------------------------------------
# TRAINING (dipicu tombol sidebar, progres SUNGGUHAN via callback)
# ----------------------------------------------------------------------------
if train_clicked:
    with st.status("Melatih DQN Agent...", expanded=True) as status:
        st.write("Mengambil matriks jarak & durasi riil dari OSRM...")
        routing_data = dl.get_distance_matrix(points)
        st.session_state.routing_data = routing_data

        if routing_data.source == "haversine_fallback":
            st.warning(
                "OSRM tidak terjangkau -- menggunakan estimasi jarak garis lurus (haversine) "
                "sebagai fallback. Hasil tetap valid untuk demo, namun kurang presisi "
                "dibanding rute jalan riil."
            )
        else:
            st.write("✅ Matriks jarak riil OSRM berhasil diambil.")

        progress_bar = st.progress(0.0)
        metrics_placeholder = st.empty()

        def on_progress(ep, total, reward, loss):
            progress_bar.progress(ep / total)
            if ep % max(1, total // 20) == 0 or ep == total:
                metrics_placeholder.write(f"Episode {ep}/{total} -- reward: {reward:.1f} | loss: {loss:.4f}")

        st.write("Menyusun grup kendaraan (klasterisasi geografis)...")
        vehicle_groups = sim.assign_customers_to_vehicles(points, num_vehicles)
        st.session_state.vehicle_groups = vehicle_groups

        # Kapasitas jaringan (max_group_size) HARUS >= grup kendaraan terbesar
        # + 1 (depot), atau padding state/mask di dqn_agent akan gagal (state
        # env lebih besar dari ukuran tensor jaringan). Nilai dari slider
        # sidebar dipakai sebagai MINIMUM; dinaikkan otomatis jika perlu,
        # supaya kombinasi top_n kecamatan besar + kendaraan sedikit (mis. 1
        # kendaraan menampung 20 kecamatan) tidak membuat aplikasi crash.
        required_group_size = max((len(g) for g in vehicle_groups), default=0) + 1
        effective_max_group_size = max(st.session_state.max_group_size, required_group_size)
        if effective_max_group_size > st.session_state.max_group_size:
            st.info(
                f"Kapasitas jaringan dinaikkan otomatis dari {st.session_state.max_group_size} "
                f"menjadi {effective_max_group_size} agar muat grup kendaraan terbesar "
                f"({required_group_size - 1} kecamatan + 1 depot)."
            )
        st.session_state.max_group_size = effective_max_group_size

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
        routes = sim.build_routes_with_agent(
            agent, routing_data.distance_matrix, depot_idx, vehicle_groups, effective_max_group_size
        )
        st.session_state.routes = routes

        st.write("Mengambil geometri jalan riil (polyline) tiap rute dari OSRM...")
        route_geoms = [dl.build_full_route_geometry(points, r) for r in routes]
        st.session_state.route_geometries = route_geoms

        st.session_state.sim_states = [
            sim.VehicleSimState(vehicle_id=i, route=routes[i], path_coords=route_geoms[i])
            for i in range(len(routes))
        ]

        status.update(label="✅ Training & penyusunan rute selesai!", state="complete", expanded=False)


# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_map, tab_train, tab_data = st.tabs(["🗺️ Peta & Simulasi", "🧠 Training DQN", "📊 Data"])


# ---- TAB: DATA ----
with tab_data:
    st.subheader("Titik Pengiriman")
    st.dataframe(points, use_container_width=True)

    if st.session_state.order_build_info is not None:
        info = st.session_state.order_build_info
        st.subheader("Konteks Order Log (sumber titik di atas)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Kota", info["city"])
        c2.metric("Total Order (filter)", f"{info['summary'].n_orders:,}")
        c3.metric("Tingkat Ontime", f"{info['summary'].ontime_rate * 100:.1f}%")
        st.caption(
            f"Rentang tanggal: {info['date_from'].date()} s/d {info['date_to'].date()} | "
            f"demand tiap titik = jumlah order pada kecamatan tsb dalam rentang ini."
        )
        st.dataframe(
            info["agg"].rename(columns={"district": "Kecamatan", "demand": "Jumlah Order", "delayed": "Delayed"}),
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.routing_data is not None:
        src = st.session_state.routing_data.source
        label = "OSRM (rute jalan riil)" if src == "osrm" else "Haversine (fallback, garis lurus)"
        st.info(f"Sumber matriks jarak saat ini: **{label}**")

    preview_map = utils.build_base_map(points)
    st_folium(preview_map, use_container_width=True, height=420, key="data_preview_map")


# ---- TAB: TRAINING ----
with tab_train:
    st.subheader("Kurva Training DQN")
    history = st.session_state.training_history

    if history is None:
        st.info("Belum ada model terlatih. Klik **Latih Model DQN** di sidebar untuk memulai.")
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


# ---- TAB: PETA & SIMULASI ----
with tab_map:
    if st.session_state.routes is None:
        st.info("Latih model dan tunggu penyusunan rute selesai untuk melihat peta simulasi.")
        st_folium(utils.build_base_map(points), use_container_width=True, height=500, key="idle_map")
    else:
        routes = st.session_state.routes
        route_geoms = st.session_state.route_geometries
        routing_data = st.session_state.routing_data

        st.markdown(utils.legend_html(len(routes)), unsafe_allow_html=True)

        # --- Metrik ringkas (dibandingkan baseline nearest-neighbor) ---
        dqn_total_dist = sum(sim.route_total_distance(routing_data.distance_matrix, r) for r in routes)
        baseline_routes = [
            sim.nearest_neighbor_baseline(routing_data.distance_matrix, depot_idx, g)
            for g in st.session_state.vehicle_groups
        ]
        baseline_total_dist = sum(sim.route_total_distance(routing_data.distance_matrix, r) for r in baseline_routes)
        improvement_pct = (
            100 * (baseline_total_dist - dqn_total_dist) / baseline_total_dist if baseline_total_dist > 0 else 0
        )
        dqn_total_time_min = sum(sim.route_total_duration(routing_data.duration_matrix, r) for r in routes) / 60

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Jarak (DQN)", f"{dqn_total_dist/1000:.2f} km")
        m2.metric(
            "Efisiensi vs Nearest-Neighbor",
            f"{improvement_pct:+.1f}%",
            delta=f"{(baseline_total_dist - dqn_total_dist)/1000:+.2f} km",
            delta_color="normal" if improvement_pct >= 0 else "inverse",
        )
        m3.metric("Estimasi Total Waktu Tempuh", f"{dqn_total_time_min:.0f} menit")

        map_placeholder = st.empty()
        status_placeholder = st.empty()

        def render_static_map():
            fmap = utils.build_base_map(points)
            for i, route in enumerate(routes):
                delivered = st.session_state.sim_states[i].delivered if st.session_state.sim_states else []
                pos = (
                    st.session_state.sim_states[i].current_position
                    if st.session_state.sim_states
                    else None
                )
                utils.add_route_to_map(fmap, points, i, route_geoms[i], delivered_nodes=delivered, vehicle_position=pos)
            with map_placeholder:
                st_folium(fmap, use_container_width=True, height=520, key=f"map_{time.time()}")

        def render_status_table():
            rows = []
            for vs in st.session_state.sim_states:
                total_stops = len(vs.route) - 2  # kurangi 2 depot (awal & akhir)
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

        # --- Render awal (statis) sebelum simulasi dijalankan ---
        render_static_map()
        render_status_table()

        # --- Loop simulasi real-time, dipicu tombol "Mulai Simulasi" di sidebar ---
        if sim_clicked:
            st.session_state.sim_running = True
            # reset progres tiap kendaraan agar simulasi mulai dari awal
            st.session_state.sim_states = [
                sim.VehicleSimState(vehicle_id=i, route=routes[i], path_coords=route_geoms[i])
                for i in range(len(routes))
            ]

            max_len = max((len(vs.path_coords) for vs in st.session_state.sim_states), default=1)
            step_size = max(1, max_len // 40)  # ~40 frame animasi agar tidak terlalu lambat/cepat

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

            st.session_state.sim_running = False
            st.success("Simulasi selesai -- seluruh kendaraan telah menyelesaikan rutenya.")
