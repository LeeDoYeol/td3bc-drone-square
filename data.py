"""동료 제공 데이터셋(merged: square+triangle+circle) → TD3+BC 리플레이 버퍼.

CSV 열(23):
  episode_id, step,
  tx-x, ty-y, tz-z,          목표까지 위치오차 (3)
  qx, qy, qz, qw,            자세 쿼터니언 (4)
  vx, vy, vz,                선속도 (3)
  wx, wy, wz,                각속도 (3)
  lx, ly, lz,                lookahead 목표오차 (경로 방향 힌트) (3)
  ax, ay, az,                행동 (3)
  reward,                    = -‖목표오차‖
  done                       True/False (에피소드 마지막)

관측 s = 16차원(위 pre-action 컬럼 전부), 행동 a = 3차원(ax,ay,az).
next_state 는 같은 에피소드의 다음 행 s (done이면 자기복사)로 만든다.
"""
import os
import glob
import numpy as np
import pandas as pd
import torch

S_COLS = ["tx-x", "ty-y", "tz-z", "qx", "qy", "qz", "qw",
          "vx", "vy", "vz", "wx", "wy", "wz", "lx", "ly", "lz"]   # 관측 s (16)
A_COLS = ["ax", "ay", "az"]                                       # 행동 a (3)
STATE_DIM = len(S_COLS)     # 16
ACTION_DIM = len(A_COLS)    # 3


def resolve_csvs(data_path):
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    else:
        files = [data_path]
    if not files:
        raise FileNotFoundError(f"CSV를 찾을 수 없음: {data_path}")
    return files


def _to_done(series):
    return (series.astype(str).str.lower() == "true").to_numpy(np.float32)


def load_dataset(data_path, max_rows=None):
    """CSV(들) → (state, action, next_state, reward, not_done) numpy 배열 + max_action."""
    S, A, NS, R, ND = [], [], [], [], []
    total = 0
    for f in resolve_csvs(data_path):
        df = pd.read_csv(f)
        state = df[S_COLS].to_numpy(np.float32)
        action = df[A_COLS].to_numpy(np.float32)
        reward = df["reward"].to_numpy(np.float32).reshape(-1, 1)
        done = _to_done(df["done"]).reshape(-1, 1)
        # next_state: 한 칸 shift, 에피소드 끝(done=1)은 자기복사
        ns = np.vstack([state[1:], state[-1:]])
        ns[done[:, 0] == 1] = state[done[:, 0] == 1]

        S.append(state); A.append(action); NS.append(ns)
        R.append(reward); ND.append(1.0 - done)
        total += len(df)
        if max_rows and total >= max_rows:
            break

    state = np.concatenate(S); action = np.concatenate(A)
    next_state = np.concatenate(NS); reward = np.concatenate(R); not_done = np.concatenate(ND)
    if max_rows:
        state, action, next_state, reward, not_done = (
            x[:max_rows] for x in (state, action, next_state, reward, not_done))

    max_action = float(np.abs(action).max())
    return state, action, next_state, reward, not_done, max_action


def normalize_states(state, next_state, eps=1e-3):
    mean = state.mean(0, keepdims=True)
    std = state.std(0, keepdims=True) + eps
    return (state - mean) / std, (next_state - mean) / std, mean, std


class ReplayBuffer:
    """전체 데이터를 텐서로 보관하고 미니배치 무작위 샘플.

    obs_noise>0 : 관측(state)에 가우시안 노이즈를 더함(후처리 증강).
      ※ 진짜 '복구' 데이터가 아니라 관측 강인성용 정규화임 — 재시뮬레이션 없이 만든 근사.
    """
    def __init__(self, state, action, next_state, reward, not_done, device, obs_noise=0.0):
        self.device = device
        self.obs_noise = obs_noise
        self.state = torch.as_tensor(state, dtype=torch.float32)
        self.action = torch.as_tensor(action, dtype=torch.float32)
        self.next_state = torch.as_tensor(next_state, dtype=torch.float32)
        self.reward = torch.as_tensor(reward, dtype=torch.float32)
        self.not_done = torch.as_tensor(not_done, dtype=torch.float32)
        self.size = self.state.shape[0]

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        s = self.state[idx].to(self.device)
        ns = self.next_state[idx].to(self.device)
        if self.obs_noise > 0.0:
            s = s + self.obs_noise * torch.randn_like(s)
            ns = ns + self.obs_noise * torch.randn_like(ns)
        return (s, self.action[idx].to(self.device), ns,
                self.reward[idx].to(self.device), self.not_done[idx].to(self.device))
