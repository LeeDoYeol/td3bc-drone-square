"""오프라인 평가 (시뮬레이터 불필요).

학습에 쓰지 않은(held-out) 데이터에서 정책을 평가한다:
  1) 행동 예측오차: 정책이 예측한 행동 vs 전문가 행동 MSE / 상관계수
  2) 궤적 비교: 속도를 적분해 '실제 비행 경로' vs '정책이 명령한 경로'(teacher-forced) 를 겹쳐 그림
     (동료 시뮬레이터가 없어 닫힌루프 롤아웃 대신 오프라인 비교)

    python eval_offline.py --data data/merged1.5M.csv --model runs/td3bc_model.pt \
        --norm runs/norm.npz --episode 200
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import S_COLS, A_COLS, STATE_DIM, ACTION_DIM
from td3bc import TD3_BC

DT = 0.01  # 100 Hz


def main():
    ap = argparse.ArgumentParser(description="오프라인 평가 (시뮬레이터 없이)")
    ap.add_argument("--data", default="data/merged1.5M.csv.gz")
    ap.add_argument("--model", default="runs/td3bc_model.pt")
    ap.add_argument("--norm", default="runs/norm.npz")
    ap.add_argument("--episode", type=int, default=200, help="궤적 비교용 held-out 에피소드 id")
    ap.add_argument("--sample", type=int, default=100000, help="MSE 계산용 held-out 표본 행수(뒤에서)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nz = np.load(args.norm)
    mean, std = nz["mean"], nz["std"]
    max_action = float(nz["max_action"]) if "max_action" in nz else 1.4
    agent = TD3_BC(STATE_DIM, ACTION_DIM, max_action=max_action, device=args.device)
    agent.load(args.model, map_location=args.device)

    def predict(states):
        s = ((states - mean) / std).astype(np.float32)
        with torch.no_grad():
            return agent.actor(torch.as_tensor(s, device=args.device)).cpu().numpy()

    df = pd.read_csv(args.data)

    # 1) held-out 표본 행동 MSE
    tail = df.tail(args.sample)
    S = tail[S_COLS].to_numpy(np.float32)
    A = tail[A_COLS].to_numpy(np.float32)
    P = predict(S)
    mse = float(((P - A) ** 2).mean())
    cors = [float(np.corrcoef(P[:, i], A[:, i])[0, 1]) for i in range(3)]
    print(f"[held-out n={len(tail):,}] action MSE={mse:.4f}  "
          f"corr(ax,ay,az)={[round(c, 3) for c in cors]}")

    # 2) 한 에피소드 궤적 비교 (속도 적분)
    ep = df[df["episode_id"] == args.episode]
    if len(ep) == 0:
        print(f"episode {args.episode} 없음"); return
    Se = ep[S_COLS].to_numpy(np.float32)
    Ae = ep[A_COLS].to_numpy(np.float32)
    vel = ep[["vx", "vy", "vz"]].to_numpy(np.float32)
    Pe = predict(Se)
    ep_mse = float(((Pe - Ae) ** 2).mean())

    traj_actual = np.cumsum(vel * DT, axis=0)   # 실제 비행(달성 속도 적분)
    traj_policy = np.cumsum(Pe * DT, axis=0)     # 정책 명령(예측 target velocity 적분)

    plt.figure(figsize=(6, 6))
    plt.plot(traj_actual[:, 0], traj_actual[:, 1], "k-", lw=1.5, label="actual (flown)")
    plt.plot(traj_policy[:, 0], traj_policy[:, 1], "b-", lw=1.2, label="policy (predicted)")
    plt.axis("equal"); plt.grid(True, alpha=0.3); plt.legend()
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.title(f"episode {args.episode} (held-out)  action MSE={ep_mse:.4f}")
    out = os.path.join(args.out, f"offline_eval_ep{args.episode}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"[episode {args.episode}] action MSE={ep_mse:.4f}  plot -> {out}")


if __name__ == "__main__":
    main()
