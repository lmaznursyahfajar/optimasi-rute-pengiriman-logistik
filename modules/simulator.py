"""
modules/simulator.py
----------------------
Menjembatani hasil DQN (urutan kunjungan per grup) dengan kebutuhan
simulasi visual: klasterisasi pelanggan ke kendaraan, metrik efisiensi,
baseline pembanding (nearest-neighbor heuristic), dan state simulasi
untuk animasi posisi kendaraan berjalan di sepanjang polyline rute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import dqn_agent as dqn


def assign_customers_to_vehicles(points: pd.DataFrame, num_vehicles: int) -> list[list[int]]:
    """Klasterisasi K-Means sederhana berdasarkan lat/lon untuk membagi
    pelanggan ke tiap kendaraan. Ini bukan bagian dari DQN -- pemisahan
    tugas yang disengaja: klasterisasi geografis menentukan SIAPA dilayani
    kendaraan mana, sedangkan DQN menentukan URUTAN optimal di dalam grup
    tersebut. Menggabungkan keduanya dalam satu agent RL (full VRP-DQN)
    jauh lebih kompleks dan di luar skala wajar untuk aplikasi demo ini.

    Return: list of list indeks GLOBAL node (belum termasuk depot),
    satu list per kendaraan.
    """
    depot_idx = points.index[points["is_depot"]][0]
    customers = points.drop(index=depot_idx)

    if len(customers) <= num_vehicles:
        # Terlalu sedikit pelanggan untuk diklaster -- satu pelanggan per kendaraan.
        return [[idx] for idx in customers.index]

    coords = customers[["lat", "lon"]].to_numpy()
    kmeans = KMeans(n_clusters=num_vehicles, n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)

    groups: list[list[int]] = [[] for _ in range(num_vehicles)]
    for local_pos, global_idx in enumerate(customers.index):
        groups[labels[local_pos]].append(int(global_idx))

    return [g for g in groups if g]  # buang klaster kosong


def build_routes_with_agent(
    agent: dqn.DQNAgent,
    distance_matrix: np.ndarray,
    depot_idx: int,
    vehicle_groups: list[list[int]],
    max_group_size: int,
) -> list[list[int]]:
    """Untuk tiap grup kendaraan, susun urutan kunjungan optimal (menurut
    kebijakan greedy agent terlatih). Return: list rute, tiap rute = list
    indeks GLOBAL node termasuk depot di awal & akhir.
    """
    routes = []
    for group in vehicle_groups:
        node_indices = [depot_idx] + group
        route = dqn.greedy_route(agent, distance_matrix, node_indices, max_group_size)
        routes.append(route)
    return routes


def nearest_neighbor_baseline(distance_matrix: np.ndarray, depot_idx: int, group: list[int]) -> list[int]:
    """Heuristik klasik sebagai pembanding: selalu pilih node terdekat
    yang belum dikunjungi. Dipakai untuk menghitung metrik 'efisiensi vs
    baseline' pada UI (delta pada st.metric).
    """
    unvisited = set(group)
    route = [depot_idx]
    current = depot_idx
    while unvisited:
        nxt = min(unvisited, key=lambda j: distance_matrix[current, j])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(depot_idx)
    return route


def route_total_distance(distance_matrix: np.ndarray, route: list[int]) -> float:
    return float(sum(distance_matrix[a, b] for a, b in zip(route[:-1], route[1:])))


def route_total_duration(duration_matrix: np.ndarray, route: list[int]) -> float:
    return float(sum(duration_matrix[a, b] for a, b in zip(route[:-1], route[1:])))


@dataclass
class VehicleSimState:
    vehicle_id: int
    route: list[int]  # indeks global node, urutan kunjungan
    path_coords: list[list[float]]  # polyline lengkap (lat, lon) hasil OSRM
    delivered: list[int] = field(default_factory=list)  # node yang sudah "terkirim"
    progress_ptr: int = 0  # posisi index di path_coords
    finished: bool = False

    @property
    def current_position(self) -> list[float]:
        if not self.path_coords:
            return [0.0, 0.0]
        idx = min(self.progress_ptr, len(self.path_coords) - 1)
        return self.path_coords[idx]


def advance_vehicle(state: VehicleSimState, points: pd.DataFrame, step_size: int, proximity_thresh_idx: int = 3):
    """Majukan satu kendaraan sepanjang polyline-nya sejauh `step_size`
    titik koordinat, lalu tandai node rute yang baru saja "terlewati"
    sebagai delivered jika posisi kendaraan cukup dekat urutannya.
    """
    if state.finished or not state.path_coords:
        state.finished = True
        return

    state.progress_ptr = min(state.progress_ptr + step_size, len(state.path_coords) - 1)

    # Tandai delivered secara proporsional terhadap progres path (pendekatan
    # sederhana: bagi path menjadi segmen sejumlah leg rute).
    total_legs = max(1, len(state.route) - 1)
    progress_fraction = state.progress_ptr / max(1, len(state.path_coords) - 1)
    legs_passed = int(progress_fraction * total_legs)

    for i in range(1, legs_passed + 1):
        node = state.route[i]
        if node not in state.delivered and node != state.route[0]:
            state.delivered.append(node)

    if state.progress_ptr >= len(state.path_coords) - 1:
        state.finished = True
