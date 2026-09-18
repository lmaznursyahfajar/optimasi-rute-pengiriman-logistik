"""
modules/dqn_agent.py
---------------------
Implementasi Deep Q-Network yang SEBENARNYA (PyTorch), menggantikan
tabular Q-learning pada kode lama.

Desain masalah (disederhanakan agar benar-benar bisa dilatih & konvergen
dalam waktu wajar untuk demo Streamlit):
- Environment = varian Traveling Salesman Problem (TSP) per kendaraan.
  Agent berada di satu node, harus memilih node belum-terkunjungi
  berikutnya, reward = -jarak(current, next). Episode selesai saat semua
  node di grup kendaraan tsb sudah dikunjungi lalu kembali ke depot.
- State = concat(one-hot posisi saat ini, mask node terkunjungi).
  Representasi ini cukup kaya untuk problem berukuran kecil-menengah
  (<= ~20 node per kendaraan) tanpa perlu graph neural network yang
  jauh lebih kompleks -- trade-off yang wajar untuk skala demo/portofolio.
- Aksi tidak valid (node yang sudah dikunjungi) di-mask sebelum argmax,
  supaya agent tidak pernah "curang" memilih aksi ilegal.

Kenapa satu DQN dipakai untuk semua kendaraan?
Karena tiap kendaraan pada dasarnya menyelesaikan sub-TSP di grup
pelanggannya sendiri, satu jaringan yang dilatih pada distribusi masalah
routing serupa bisa digeneralisasi ke tiap grup (weight sharing), lebih
efisien daripada melatih agent terpisah per kendaraan.
"""

from __future__ import annotations

import random
from collections import deque, namedtuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "next_mask"])


class QNetwork(nn.Module):
    """MLP sederhana: cukup untuk state berbasis one-hot + mask pada graf kecil."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class RoutingEnv:
    """Environment TSP-like satu episode = satu rute kendaraan dari depot,
    mengunjungi semua node dalam grup, lalu kembali ke depot.
    """

    def __init__(self, distance_matrix: np.ndarray, node_indices: list[int], depot_local_idx: int = 0):
        self.full_distance_matrix = distance_matrix
        self.node_indices = node_indices  # indeks node (dlm koordinat global) yg termasuk grup ini
        self.n = len(node_indices)
        self.depot_local_idx = depot_local_idx

    def _dist(self, i_local: int, j_local: int) -> float:
        gi, gj = self.node_indices[i_local], self.node_indices[j_local]
        return float(self.full_distance_matrix[gi, gj])

    def reset(self):
        self.current = self.depot_local_idx
        self.visited = np.zeros(self.n, dtype=bool)
        self.visited[self.current] = True
        self.steps = 0
        return self._state()

    def _state(self) -> np.ndarray:
        pos_onehot = np.zeros(self.n)
        pos_onehot[self.current] = 1.0
        return np.concatenate([pos_onehot, self.visited.astype(float)])

    def valid_action_mask(self) -> np.ndarray:
        """True = aksi valid (node belum dikunjungi)."""
        mask = ~self.visited
        return mask

    def step(self, action_local: int):
        dist = self._dist(self.current, action_local)
        reward = -dist  # agent belajar minimalkan total jarak
        self.current = action_local
        self.visited[action_local] = True
        self.steps += 1

        done = bool(self.visited.all())
        if done:
            # tambahkan jarak kembali ke depot supaya rute benar-benar tertutup
            reward -= self._dist(self.current, self.depot_local_idx)

        return self._state(), reward, done, self.valid_action_mask()


@dataclass
class TrainingHistory:
    episode_rewards: list[float] = field(default_factory=list)
    episode_losses: list[float] = field(default_factory=list)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.95,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def act(self, state: np.ndarray, valid_mask: np.ndarray, epsilon: float) -> int:
        valid_indices = np.flatnonzero(valid_mask)
        if random.random() < epsilon:
            return int(random.choice(valid_indices))

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        q_values_masked = np.where(valid_mask, q_values, -np.inf)
        return int(np.argmax(q_values_masked))

    def train_step(self, batch_size: int = 64) -> float | None:
        if len(self.buffer) < batch_size:
            return None

        batch = self.buffer.sample(batch_size)
        states = torch.as_tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.array([t.next_mask for t in batch]), dtype=torch.bool, device=self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q = next_q.masked_fill(~next_masks, -torch.inf)
            next_q_max = next_q.max(dim=1).values
            next_q_max = torch.nan_to_num(next_q_max, neginf=0.0)  # episode terakhir: tidak ada next valid action
            target = rewards + self.gamma * next_q_max * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self, tau: float = 0.1):
        """Soft update supaya target network bergerak halus mengikuti policy net."""
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def train_agent(
    distance_matrix: np.ndarray,
    max_group_size: int,
    episodes: int = 300,
    lr: float = 1e-3,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 200,
    progress_callback=None,
) -> tuple[DQNAgent, TrainingHistory]:
    """Latih satu DQN agent yang digeneralisasi untuk ukuran grup <= max_group_size,
    dengan episode disampling dari sub-grup node acak setiap kalinya. Ini melatih
    agent pada variasi ukuran & susunan sub-rute, bukan hanya satu instance tetap,
    supaya lebih general saat dipakai pada grup kendaraan riil di tab simulasi.

    `progress_callback(episode, total, reward, loss)` dipanggil tiap episode agar
    UI (st.status / st.progress) bisa menampilkan progres training yang SUNGGUHAN,
    bukan animasi dekoratif yang tidak sinkron dengan proses aktual.
    """
    state_dim = 2 * max_group_size
    action_dim = max_group_size
    agent = DQNAgent(state_dim, action_dim, lr=lr, gamma=gamma)
    history = TrainingHistory()

    n_total_nodes = distance_matrix.shape[0]

    for ep in range(episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, 1 - ep / max(1, epsilon_decay_episodes)
        )

        # Sampling grup acak: depot (index 0, diasumsikan node 0 = depot global)
        # + antara 3..max_group_size-1 node pelanggan acak, agar agent belajar
        # menggeneralisasi ke berbagai ukuran & komposisi sub-rute.
        group_size = random.randint(3, max_group_size)
        customer_pool = list(range(1, n_total_nodes))
        chosen_customers = random.sample(customer_pool, min(group_size - 1, len(customer_pool)))
        node_indices = [0] + chosen_customers  # depot selalu index lokal 0

        env = RoutingEnv(distance_matrix, node_indices, depot_local_idx=0)
        state = _pad_state(env.reset(), state_dim)
        mask = _pad_mask(env.valid_action_mask(), action_dim)

        done = False
        ep_reward = 0.0
        ep_losses = []

        while not done:
            action_local = agent.act(state, mask, epsilon)
            next_state_raw, reward, done, next_mask_raw = env.step(action_local)

            next_state = _pad_state(next_state_raw, state_dim)
            next_mask = _pad_mask(next_mask_raw, action_dim)

            agent.buffer.push(state, action_local, reward, next_state, done, next_mask)

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
                agent.update_target()

            state, mask = next_state, next_mask
            ep_reward += reward

        history.episode_rewards.append(ep_reward)
        history.episode_losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if progress_callback is not None:
            progress_callback(ep + 1, episodes, ep_reward, history.episode_losses[-1])

    return agent, history


def _pad_state(state: np.ndarray, target_dim: int) -> np.ndarray:
    if len(state) == target_dim:
        return state
    padded = np.zeros(target_dim)
    padded[: len(state)] = state
    return padded


def _pad_mask(mask: np.ndarray, target_dim: int) -> np.ndarray:
    if len(mask) == target_dim:
        return mask
    padded = np.zeros(target_dim, dtype=bool)
    padded[: len(mask)] = mask
    return padded


def greedy_route(agent: DQNAgent, distance_matrix: np.ndarray, node_indices: list[int], max_group_size: int) -> list[int]:
    """Gunakan agent terlatih (epsilon=0, full-greedy) untuk menyusun urutan
    kunjungan pada satu grup kendaraan riil. Return: list indeks GLOBAL
    (bukan lokal) sesuai urutan kunjungan, termasuk kembali ke depot di akhir.
    """
    state_dim = 2 * max_group_size
    action_dim = max_group_size

    env = RoutingEnv(distance_matrix, node_indices, depot_local_idx=0)
    state = _pad_state(env.reset(), state_dim)
    mask = _pad_mask(env.valid_action_mask(), action_dim)

    route_local = [0]
    done = False
    while not done:
        action_local = agent.act(state, mask, epsilon=0.0)
        state_raw, _, done, mask_raw = env.step(action_local)
        route_local.append(action_local)
        state = _pad_state(state_raw, state_dim)
        mask = _pad_mask(mask_raw, action_dim)

    route_local.append(0)  # kembali ke depot
    return [node_indices[i] for i in route_local]
