"""
modules/data_loader.py
-----------------------
Modul ini HANYA bertanggung jawab atas ROUTING (OSRM), bukan sumber data
titik pengiriman -- itu tugas modules/olist_data.py.

1. Menghitung matriks jarak & durasi riil antar titik lewat OSRM (Open
   Source Routing Machine), dengan fallback ke perhitungan haversine
   (geopy) jika OSRM tidak bisa diakses (offline, rate-limit, dsb).
2. Mengambil geometri rute riil (polyline) antar dua titik untuk digambar
   di peta -- bukan sekadar garis lurus antar marker.

Kenapa OSRM demo server?
Server publik router.project-osrm.org gratis, tidak butuh API key, dan
cukup untuk keperluan portofolio/demo, serta memiliki cakupan jalan raya
global (termasuk Brasil). Untuk produksi, ganti OSRM_BASE_URL dengan
instance OSRM self-hosted agar tidak kena rate-limit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import streamlit as st
from geopy.distance import geodesic

OSRM_BASE_URL = "https://router.project-osrm.org"
ASSUMED_AVG_SPEED_KMH = 30.0  # dipakai hanya pada fallback haversine


@dataclass
class RoutingData:
    """Wadah hasil akhir tahap routing: titik, matriks jarak/durasi, dan sumbernya."""

    points: pd.DataFrame
    distance_matrix: np.ndarray  # meter
    duration_matrix: np.ndarray  # detik
    source: str  # "osrm" atau "haversine_fallback"


def _haversine_matrix(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fallback jika OSRM tidak terjangkau: jarak garis lurus (haversine),
    durasi diestimasi dari kecepatan rata-rata asumsi. Tidak sepresisi
    OSRM (yang mengikuti jaringan jalan), tapi menjaga aplikasi tetap
    berjalan saat offline / API down.
    """
    n = len(points)
    dist_m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            coord_i = (points.iloc[i]["lat"], points.iloc[i]["lon"])
            coord_j = (points.iloc[j]["lat"], points.iloc[j]["lon"])
            dist_m[i, j] = geodesic(coord_i, coord_j).meters
    duration_s = dist_m / (ASSUMED_AVG_SPEED_KMH * 1000 / 3600)
    return dist_m, duration_s


@st.cache_data(show_spinner=False, ttl=3600)
def get_distance_matrix(points: pd.DataFrame) -> RoutingData:
    """Ambil matriks jarak & durasi riil via OSRM Table Service.

    OSRM /table mengembalikan matriks NxN sekaligus dalam satu request,
    jauh lebih efisien daripada memanggil /route satu-per-satu (N^2 calls).
    Di-cache oleh Streamlit berdasarkan isi `points` supaya tidak memanggil
    API berulang kali setiap kali widget lain berubah.
    """
    coords = ";".join(f"{row.lon},{row.lat}" for _, row in points.iterrows())
    url = f"{OSRM_BASE_URL}/table/v1/driving/{coords}"
    params = {"annotations": "distance,duration"}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok":
            raise RuntimeError(data.get("message", "OSRM mengembalikan status non-Ok"))

        dist = np.array(data["distances"], dtype=float)
        dur = np.array(data["durations"], dtype=float)

        if np.isnan(dist).any() or np.isnan(dur).any():
            raise RuntimeError("Sebagian pasangan titik tidak terjangkau oleh OSRM.")

        return RoutingData(points=points, distance_matrix=dist, duration_matrix=dur, source="osrm")

    except Exception:
        # Fallback senyap ke haversine -- UI yang memanggil fungsi ini
        # bertanggung jawab menampilkan `source` ke pengguna agar transparan.
        dist, dur = _haversine_matrix(points)
        return RoutingData(points=points, distance_matrix=dist, duration_matrix=dur, source="haversine_fallback")


@st.cache_data(show_spinner=False, ttl=3600)
def get_route_geometry(lat1: float, lon1: float, lat2: float, lon2: float) -> list[list[float]]:
    """Ambil polyline riil (mengikuti jalan) antar dua titik via OSRM /route.

    Return: list koordinat [lat, lon] siap dipakai folium.PolyLine.
    Fallback: garis lurus 2 titik jika OSRM gagal, supaya peta tetap tergambar.
    """
    url = f"{OSRM_BASE_URL}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "full", "geometries": "geojson"}
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        coords = data["routes"][0]["geometry"]["coordinates"]  # [lon, lat] pairs
        return [[lat, lon] for lon, lat in coords]
    except Exception:
        return [[lat1, lon1], [lat2, lon2]]


def build_full_route_geometry(points: pd.DataFrame, route_indices: list[int]) -> list[list[float]]:
    """Sambungkan geometri OSRM antar setiap leg dalam satu rute kendaraan
    menjadi satu polyline panjang, agar digambar sebagai satu garis rute utuh.
    """
    full_path: list[list[float]] = []
    for a, b in zip(route_indices[:-1], route_indices[1:]):
        p1, p2 = points.iloc[a], points.iloc[b]
        leg = get_route_geometry(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
        full_path.extend(leg if not full_path else leg[1:])
        time.sleep(0.05)  # sopan terhadap rate-limit server demo OSRM
    return full_path
