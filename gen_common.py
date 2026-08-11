"""생성모델(diffusion/GAN) 공용: 전이(transition)를 하나의 벡터로 다루는 유틸.

SynthER 방식 — 전이 하나를 통째로 생성한다:
    x = [ s(16) | a(3) | s'(16) | r(1) ]   (36차원)

생성된 x 는 실제 데이터와 같은 원 단위(raw)로 되돌려 train.py 의 리플레이 버퍼에
그대로 섞어 넣는다(정규화는 train.py 가 실데이터+합성데이터 전체에 대해 다시 수행).

not_done: 합성 전이는 에피소드 중간 전이로 간주해 1(=종료 아님)로 둔다.
실제 데이터에서 done=1 인 행은 전체의 0.03% 수준(에피소드당 1행)이라 무시해도 무방.
"""
import numpy as np
import torch
import torch.nn as nn

from data import STATE_DIM, ACTION_DIM

S0, S1 = 0, STATE_DIM                              # s
A0, A1 = S1, S1 + ACTION_DIM                       # a
N0, N1 = A1, A1 + STATE_DIM                        # s'
R0, R1 = N1, N1 + 1                                # r
DIM = R1                                           # 36

#### 상태 16열 중 자세 쿼터니언 (qx,qy,qz,qw) 위치 — 물리 제약(단위 노름) 복원용
QUAT = slice(3, 7)


def build_matrix(state, action, next_state, reward, delta=False):
    """(N,16),(N,3),(N,16),(N,1) → (N,36) 전이 행렬.

    delta=True 면 s' 자리에 변화량 (s'-s) 을 넣는다. 한 스텝(10ms) 동안 상태는 아주 조금만
    변하므로 s 와 s' 는 거의 같은 값이다 — 절대값으로 두 번 생성하게 하면 생성모델이 그
    강한 결합을 놓쳐 물리적으로 불가능한 전이를 만든다(동역학 잔차 30~60배). 변화량으로
    두면 그 결합이 표현 자체에 내장되어 훨씬 쉽게 학습된다.
    """
    tgt = (next_state - state) if delta else next_state
    return np.concatenate([state, action, tgt, reward.reshape(-1, 1)], axis=1).astype(np.float32)


def split_matrix(x, delta=False):
    """(N,36) → (state, action, next_state, reward) — build_matrix 의 역변환."""
    s, a, third, r = x[:, S0:S1], x[:, A0:A1], x[:, N0:N1], x[:, R0:R1]
    return s, a, (s + third) if delta else third, r


def fit_norm(x):
    """생성모델 학습용 per-dim 표준화 통계."""
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def postprocess(x, lo, hi):
    """실데이터의 per-dim 범위로 클리핑(범위 밖 헛값 제거).

    행동 a 가 실제 max_action 을 넘지 않도록 하는 역할도 겸한다.
    """
    return np.clip(x, lo, hi)


def _unit_quat(s):
    """자세 쿼터니언을 단위 노름으로 복원(생성모델은 이 제약을 모른다)."""
    q = s[:, QUAT]
    n = np.linalg.norm(q, axis=1, keepdims=True)
    s[:, QUAT] = np.where(n > 1e-6, q / np.maximum(n, 1e-6), np.array([0, 0, 0, 1.0], np.float32))
    return s


def save_synth(path, x, delta=False):
    """(N,36) → train.py --extra_data 가 읽는 npz."""
    s, a, ns, r = split_matrix(x, delta)
    s, ns = _unit_quat(s.copy()), _unit_quat(ns.copy())
    np.savez_compressed(path, state=s, action=a, next_state=ns, reward=r,
                        not_done=np.ones((len(x), 1), np.float32))
    print(f"[OK] synthetic {len(x):,} transitions -> {path}")


def load_synth(path):
    z = np.load(path)
    return (z["state"], z["action"], z["next_state"], z["reward"], z["not_done"])


def real_matrix(data_path, max_rows=None, delta=False):
    """실제 CSV → (X(N,36), lo, hi) — 생성모델 학습 데이터와 클리핑 범위."""
    from data import load_dataset
    state, action, next_state, reward, _, max_action = load_dataset(data_path, max_rows)
    x = build_matrix(state, action, next_state, reward, delta)
    lo, hi = x.min(0, keepdims=True), x.max(0, keepdims=True)
    print(f"real transitions={len(x):,} dim={x.shape[1]} (max_action={max_action:.3f})")
    return x, lo, hi


class MLP(nn.Module):
    """생성모델 공용 MLP 몸통 (diffusion 잡음예측기 / GAN 생성기·판별기)."""

    def __init__(self, in_dim, out_dim, hidden=512, depth=3, act=nn.SiLU):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), act()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def batches(n, batch_size, generator=None):
    """무작위 인덱스 미니배치 생성기(무한)."""
    while True:
        yield torch.randint(0, n, (batch_size,), generator=generator)
