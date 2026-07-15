"""학습된 TD3+BC 정책을 데이터셋을 만든 것과 같은 시뮬레이터로 롤아웃 평가.

정책을 policy_fn 으로 감싸 shape_dataset.run 에 넘긴다. obs/행동 규약은 데이터 생성과 동일:
  obs(16)  = [pos_err(3), quat(4), vel(3), angvel(3), lookahead(3)]
  action(3)= target velocity (m/s), 데이터의 max_accel(=2.0)과 같은 slew-rate 제한 재적용
여러 도형(직선/삼각형/사각형/원/별)을 held-out seed 로 롤아웃해, 전문가 대비 추적오차를
출력하고 모든 도형을 한 그림에 모아 저장한다. 목표 경로는 같은 seed로 전체를 재생성해 그린다.

    KMP_DUPLICATE_LIB_OK=TRUE python sim_eval.py --run_dir runs --seed 500
"""
import os
import math
import argparse
import csv as csvmod
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shape_dataset as sd
from td3bc import TD3_BC

S_DIM, A_DIM = 16, 3


def load_agent(run_dir, device="cpu"):
    nz = np.load(os.path.join(run_dir, "norm.npz"))
    mean, std = nz["mean"].astype(np.float32).reshape(-1), nz["std"].astype(np.float32).reshape(-1)
    max_action = float(nz["max_action"]) if "max_action" in nz else 1.4
    agent = TD3_BC(S_DIM, A_DIM, max_action=max_action, device=device)
    agent.load(os.path.join(run_dir, "td3bc_model.pt"), map_location=device)
    return agent, mean, std


def target_path(shape, seed):
    """shape_dataset.run 과 같은 seed·기본값으로 전체 목표 경로(완성된 도형)를 재생성.

    run() 의 rng 소비 순서(sample_episode_params → generate_local_shape_waypoints)를 그대로
    따르므로 실제 롤아웃이 따라간 경로와 동일하다(기본 max_speed==max_accel 이라 그 앞 draw 없음).
    """
    rng = np.random.default_rng(seed)
    ep = sd.sample_episode_params(shape, sd.DEFAULT_RADIUS, sd.DEFAULT_SIDE_JITTER,
                                  sd.DEFAULT_TILT_MAX_DEG, sd.DEFAULT_WORKSPACE_SIZE,
                                  sd.DEFAULT_FLOOR_CLEARANCE, rng)
    local = sd.generate_local_shape_waypoints(shape, sd.DEFAULT_PATH_RESOLUTION,
                                              sd.DEFAULT_RADIUS, sd.DEFAULT_SIDE_JITTER, rng)
    return sd.place_waypoints(local, start_yaw=np.radians(ep["start_yaw_deg"]),
                              tilt_deg=np.radians(ep["tilt_deg"]),
                              tilt_axis_deg=np.radians(ep["tilt_axis_deg"]), center=ep["center"])


def make_policy_fn(agent, mean, std, slew_max_accel=2.0, ctrl_hz=100):
    max_dv = None if slew_max_accel is None else slew_max_accel / ctrl_hz
    prev = np.zeros(3)
    flown = []

    def fn(pos_err, state, lookahead=None):
        nonlocal prev
        flown.append(state[0:3].copy())
        obs = np.concatenate([pos_err, state[3:7], state[10:13], state[13:16],
                              np.asarray(lookahead, dtype=np.float32)]).astype(np.float32)
        obs = (obs - mean) / std
        with torch.no_grad():
            raw = agent.actor(torch.from_numpy(obs)).cpu().numpy()
        act = raw
        if max_dv is not None:                       # 데이터와 동일한 slew 제한 재적용
            d = raw - prev
            dm = np.linalg.norm(d)
            if dm > max_dv:
                act = prev + d * (max_dv / dm)
            prev = act
        return act
    return fn, flown


def latest_csv(folder):
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    return max(fs, key=os.path.getmtime)


def track_err(csv_path):
    rows = list(csvmod.DictReader(open(csv_path)))
    pe = np.array([[float(r["tx-x"]), float(r["ty-y"]), float(r["tz-z"])] for r in rows])
    return float(np.linalg.norm(pe, axis=1).mean())


def combined_plot(results, out_png):
    """results: [(shape, target(N,3), flown(M,3), e_mean, p_mean), ...] → 한 그림에."""
    n = len(results)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(5 * ncols, 4.5 * nrows))
    for i, (shape, target, flown, e, p) in enumerate(results):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        target = np.array(target); flown = np.array(flown)
        ax.plot(target[:, 0], target[:, 1], target[:, 2], "k--", lw=1, label="target")
        ax.plot(flown[:, 0], flown[:, 1], flown[:, 2], "b-", lw=1.1, label="policy")
        ax.scatter(*flown[0], c="g", s=20)
        allp = np.vstack([target, flown])
        mid = (allp.min(0) + allp.max(0)) / 2
        hr = max((allp.max(0) - allp.min(0)).max(), 1e-6) / 2
        ax.set_xlim(mid[0]-hr, mid[0]+hr); ax.set_ylim(mid[1]-hr, mid[1]+hr); ax.set_zlim(mid[2]-hr, mid[2]+hr)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"{shape}\npolicy {p:.2f}m (expert {e:.2f}m)", fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("same-simulator rollout: target vs TD3+BC policy", fontsize=12)
    fig.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close()


def main():
    ap = argparse.ArgumentParser(description="같은 시뮬레이터 롤아웃 평가")
    ap.add_argument("--run_dir", default="runs", help="norm.npz + td3bc_model.pt 폴더")
    ap.add_argument("--shapes", nargs="+",
                    default=["line", "triangle", "square", "circle", "star"])
    ap.add_argument("--seed", type=int, default=500, help="held-out seed")
    ap.add_argument("--slew", type=float, default=2.0, help="slew max accel[m/s^2] (데이터=2.0)")
    ap.add_argument("--out", default="sim_eval_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    agent, mean, std = load_agent(args.run_dir)
    print(f"{'shape':<10} {'expert_err':>11} {'policy_err':>11}  (mean tracking error, m)")
    results = []
    for shape in args.shapes:
        sd.run(shape=shape, seed=args.seed, gui=False, att_d_gain_scale=0.3,
               output_folder=os.path.join(args.out, "expert"))
        pf, flown = make_policy_fn(agent, mean, std, slew_max_accel=args.slew)
        tgt_capture = []   # 시뮬레이터가 실제로 쓴 전체 목표 경로를 직접 받음(항상 완전)
        sd.run(shape=shape, seed=args.seed, gui=False, att_d_gain_scale=0.3,
               output_folder=os.path.join(args.out, "policy"), policy_fn=pf, target_out=tgt_capture)
        e_mean = track_err(latest_csv(os.path.join(args.out, "expert", "shape_dataset")))
        p_mean = track_err(latest_csv(os.path.join(args.out, "policy", "shape_dataset")))
        target = tgt_capture[0] if tgt_capture else target_path(shape, args.seed)
        results.append((shape, target, flown, e_mean, p_mean))
        print(f"{shape:<10} {e_mean:>11.4f} {p_mean:>11.4f}")

    png = os.path.join(args.out, "trajectories_all.png")
    combined_plot(results, png)
    print(f"combined plot -> {png}")


if __name__ == "__main__":
    main()
