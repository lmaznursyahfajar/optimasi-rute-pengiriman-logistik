"""
modules/utils.py
------------------
Helper murni untuk presentasi: membangun peta Folium dengan polyline rute
riil (bukan garis lurus antar marker) dan warna berbeda tiap kendaraan,
plus penanda posisi kendaraan saat simulasi berjalan.
"""

from __future__ import annotations

import folium
import pandas as pd

VEHICLE_COLORS = [
    "#E63946", "#2A9D8F", "#457B9D", "#F4A261", "#8338EC",
    "#FB5607", "#3A86FF", "#06D6A0", "#EF476F", "#118AB2",
]


def vehicle_color(vehicle_id: int) -> str:
    return VEHICLE_COLORS[vehicle_id % len(VEHICLE_COLORS)]


def build_base_map(points: pd.DataFrame) -> folium.Map:
    depot = points[points["is_depot"]].iloc[0]
    fmap = folium.Map(location=[depot["lat"], depot["lon"]], zoom_start=12, tiles="cartodbpositron")

    folium.Marker(
        [depot["lat"], depot["lon"]],
        popup="Depot Utama",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(fmap)

    for _, row in points[~points["is_depot"]].iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]],
            radius=5,
            popup=f"{row['name']} (demand: {row['demand']})",
            color="#495057",
            fill=True,
            fill_opacity=0.7,
        ).add_to(fmap)

    return fmap


def add_route_to_map(
    fmap: folium.Map,
    points: pd.DataFrame,
    vehicle_id: int,
    path_coords: list[list[float]],
    delivered_nodes: list[int] | None = None,
    vehicle_position: list[float] | None = None,
) -> folium.Map:
    """Gambar satu rute kendaraan sebagai polyline riil di atas map yang
    sudah ada, dengan warna khas kendaraan tsb. Opsional: tandai node yang
    sudah delivered dan posisi kendaraan saat ini (untuk mode simulasi).
    """
    color = vehicle_color(vehicle_id)

    if path_coords:
        folium.PolyLine(
            path_coords,
            color=color,
            weight=4,
            opacity=0.8,
            tooltip=f"Kendaraan #{vehicle_id + 1}",
        ).add_to(fmap)

    delivered_nodes = delivered_nodes or []
    for node_idx in delivered_nodes:
        row = points.loc[node_idx]
        folium.CircleMarker(
            [row["lat"], row["lon"]],
            radius=7,
            color=color,
            fill=True,
            fill_color="#2ECC71",
            fill_opacity=1.0,
            popup=f"✅ Terkirim: {row['name']}",
        ).add_to(fmap)

    if vehicle_position is not None:
        folium.Marker(
            vehicle_position,
            icon=folium.DivIcon(
                html=f"""
                <div style="font-size:22px; transform: translate(-10px, -10px);
                            filter: drop-shadow(0 0 2px white);">
                    <span style="color:{color};">🚚</span>
                </div>"""
            ),
            popup=f"Kendaraan #{vehicle_id + 1}",
        ).add_to(fmap)

    return fmap


def legend_html(num_vehicles: int) -> str:
    """HTML kecil untuk legenda warna kendaraan, ditampilkan via st.markdown."""
    items = "".join(
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{vehicle_color(i)};border-radius:3px;margin-right:6px;"></span>'
        f'Kendaraan #{i + 1}&nbsp;&nbsp;&nbsp;'
        for i in range(num_vehicles)
    )
    return f'<div style="padding:6px 0;font-size:14px;">{items}</div>'
