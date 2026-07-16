"""시작점 '빙글빙글' 원인 조사: 전문가(pure-pursuit) vs 학습 정책의 시작부 궤적 비교.

전문가도 정지->출발 과도로 시작점에서 도는지 확인하면, 그 회전이 데이터에서 물려받은
것인지(정책 버그 아님) 아니면 정책 고유 문제인지 구분할 수 있다.

    KMP_DUPLICATE_LIB_OK=TRUE python investigate_start.py --run_dir runs_c --shape square --seed 500
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shape_dataset as sd
import sim_eval as se
from td3bc import TD3_BC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs_c")
    ap.add_argument("--shape", default="square")
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--slew", type=float, default=2.0)
    ap.add_argument("--start_steps", type=int, default=200, help="시작부로 볼 스텝 수")
    ap.add_argument("--out", default="investigate_start")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    nz = np.load(os.path.join(args.run_dir, "norm.npz"))
    mean, std = nz["mean"].reshape(-1), nz["std"].reshape(-1)
    maxa = float(nz["max_action"]) if "max_action" in nz else 1.4

    model = os.path.join(args.run_dir, "best_model.pt")
    if not os.path.exists(model):
        model = os.path.join(args.run_dir, "td3bc_model.pt")
    agent = TD3_BC(16, 3, max_action=maxa, device="cpu")
    agent.load(model, map_location="cpu")

    # 전문가 롤아웃 (policy_fn=None → pure-pursuit)
    tgt = []
    exp_pos = []
    sd.run(shape=args.shape, seed=args.seed, gui=False, att_d_gain_scale=0.3,
           output_folder=os.path.join(args.out, "_exp"), policy_fn=None,
           target_out=tgt, pos_out=exp_pos)
    target = np.array(tgt[0]); exp_pos = np.array(exp_pos)

    # 정책 롤아웃
    pf, flown = se.make_policy_fn(agent, mean, std, slew_max_accel=args.slew)
    pol_pos = []
    sd.run(shape=args.shape, seed=args.seed, gui=False, att_d_gain_scale=0.3,
           output_folder=os.path.join(args.out, "_pol"), policy_fn=pf, pos_out=pol_pos)
    pol_pos = np.array(pol_pos)

    ns = args.start_steps
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    # 전체 (위에서 본 XY)
    ax = axs[0]
    ax.plot(target[:, 0], target[:, 1], "k--", lw=1, label="target")
    ax.plot(exp_pos[:, 0], exp_pos[:, 1], "g-", lw=1, alpha=0.7, label="expert(pursuit)")
    ax.plot(pol_pos[:, 0], pol_pos[:, 1], "b-", lw=1, alpha=0.8, label="policy")
    ax.scatter(*target[0, :2], c="r", s=40, zorder=5, label="start")
    ax.set_title(f"{args.shape}: full path (top view)")
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_xlabel("x"); ax.set_ylabel("y")
    # 시작부 확대
    ax = axs[1]
    ax.plot(target[:, 0], target[:, 1], "k--", lw=1, label="target")
    ax.plot(exp_pos[:ns, 0], exp_pos[:ns, 1], "g.-", lw=1, ms=3, label=f"expert first {ns}")
    ax.plot(pol_pos[:ns, 0], pol_pos[:ns, 1], "b.-", lw=1, ms=3, label=f"policy first {ns}")
    ax.scatter(*target[0, :2], c="r", s=40, zorder=5, label="start")
    c = target[0, :2]
    ax.set_xlim(c[0]-0.8, c[0]+0.8); ax.set_ylim(c[1]-0.8, c[1]+0.8)
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_title("START region (zoom)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(f"start-transient check: {args.shape} ({os.path.basename(args.run_dir)})")
    fig.tight_layout()
    png = os.path.join(args.out, f"start_{args.shape}.png")
    plt.savefig(png, dpi=120, bbox_inches="tight"); plt.close()

    # 시작부 이동거리/속도 요약
    def moved(p, k):
        return np.linalg.norm(np.diff(p[:k], axis=0), axis=1).sum()
    print(f"[expert] start {ns}steps path-length={moved(exp_pos, ns):.3f} m  "
          f"straight-dist={np.linalg.norm(exp_pos[ns-1]-exp_pos[0]):.3f} m")
    print(f"[policy] start {ns}steps path-length={moved(pol_pos, ns):.3f} m  "
          f"straight-dist={np.linalg.norm(pol_pos[ns-1]-pol_pos[0]):.3f} m")
    print(f"plot -> {png}")


if __name__ == "__main__":
    main()
