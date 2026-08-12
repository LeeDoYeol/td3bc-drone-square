"""저장된 체크포인트(ckpts/ckpt_*.pt)를 같은 시뮬레이터로 평가해 best를 선택.

오프라인 RL은 마지막 체크포인트가 최고가 아니므로(비단조적), 여러 체크포인트를 굴려
가장 추적오차가 낮은 것을 골라 best_model.pt 로 저장하고 그 궤적 그림을 만든다.

    KMP_DUPLICATE_LIB_OK=TRUE python select_best.py --run_dir runs --shapes square --seed 500
"""
import os
import glob
import shutil
import argparse
import numpy as np
import torch

import shape_dataset as sd
import sim_eval as se
from td3bc import TD3_BC


def eval_model(model_path, mean, std, maxa, shapes, seed, slew, out_tmp, att_d_gain=0.3,
               cov_tol=0.5):
    agent = TD3_BC(16, 3, max_action=maxa, device="cpu")
    agent.load(model_path, map_location="cpu")
    per, cov, recs = {}, {}, {}
    for shape in shapes:
        pf, flown = se.make_policy_fn(agent, mean, std, slew_max_accel=slew)
        tgt = []
        sd.run(shape=shape, seed=seed, gui=False, att_d_gain_scale=att_d_gain,
               output_folder=out_tmp, policy_fn=pf, target_out=tgt)
        per[shape] = se.track_err(se.latest_csv(os.path.join(out_tmp, "shape_dataset")))
        cov[shape] = se.path_coverage(tgt[0], np.array(flown), cov_tol)
        recs[shape] = (tgt[0], np.array(flown), per[shape])
    return (float(np.mean(list(per.values()))), float(np.min(list(cov.values()))), per, cov, recs)


def main():
    ap = argparse.ArgumentParser(description="체크포인트 best 선택")
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--shapes", nargs="+", default=["square"])
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--slew", type=float, default=2.0)
    ap.add_argument("--att_d_gain", type=float, default=0.3,
                    help="평가 시 자세 D게인 배율 — 데이터 수집 때와 같아야 함"
                         "(merged1.5M=0.3, hard_v2=1.0)")
    ap.add_argument("--cov_tol", type=float, default=0.5,
                    help="경로점에 이 거리[m] 안으로 접근하면 '지나갔다'로 센다")
    ap.add_argument("--min_coverage", type=float, default=0.9,
                    help="best 후보가 되기 위한 최소 경로 완주율(제자리 정지 정책 배제)")
    ap.add_argument("--out", default="select_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nz = np.load(os.path.join(args.run_dir, "norm.npz"))
    mean, std = nz["mean"].reshape(-1), nz["std"].reshape(-1)
    maxa = float(nz["max_action"]) if "max_action" in nz else 1.4

    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpts", "ckpt_*.pt")))
    final = os.path.join(args.run_dir, "td3bc_model.pt")
    if os.path.exists(final):
        ckpts.append(final)
    if not ckpts:
        print("체크포인트 없음. train.py --save_every 로 저장하세요."); return

    print(f"{'checkpoint':<22} {'mean':>8} {'cover':>7}  " + " ".join(f"{s:>8}" for s in args.shapes))
    rows = []
    for ck in ckpts:
        m, c, per, cov, recs = eval_model(ck, mean, std, maxa, args.shapes, args.seed, args.slew,
                                          os.path.join(args.out, "_tmp"), args.att_d_gain,
                                          args.cov_tol)
        print(f"{os.path.basename(ck):<22} {m:>8.4f} {c*100:>6.0f}%  "
              + " ".join(f"{per[s]:>8.4f}" for s in args.shapes))
        rows.append((m, c, ck, recs))

    #### 추적오차만으로 고르면 '경로 위에 가만히 떠 있는' 정책이 1등을 한다(오차≈0, 진행 없음).
    #### 그래서 먼저 모든 도형에서 경로를 min_coverage 이상 지나간 체크포인트만 남기고,
    #### 그 안에서 오차가 가장 낮은 것을 고른다. 아무도 통과 못 하면 가장 많이 돈 것을 고른다.
    ok = [r for r in rows if r[1] >= args.min_coverage]
    if ok:
        m, c, ck, recs = min(ok, key=lambda r: r[0])
    else:
        m, c, ck, recs = max(rows, key=lambda r: (r[1], -r[0]))
        print(f"\n[주의] 경로 {args.min_coverage*100:.0f}% 이상 완주한 체크포인트가 없음 "
              f"— 완주율이 가장 높은 것을 선택")
    shutil.copy(ck, os.path.join(args.run_dir, "best_model.pt"))
    print(f"\n[BEST] {os.path.basename(ck)}  mean_err={m:.4f} m  coverage={c*100:.0f}%"
          f"  -> {args.run_dir}\\best_model.pt")

    results = [(s, recs[s][0], recs[s][1], 0.0, recs[s][2]) for s in args.shapes]
    png = os.path.join(args.out, "best_trajectories.png")
    se.combined_plot(results, png)
    print(f"plot -> {png}")


if __name__ == "__main__":
    main()
