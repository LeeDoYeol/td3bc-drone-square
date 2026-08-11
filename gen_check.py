"""합성 전이(diffusion/GAN) 품질 점검 — 학습에 쓰기 전 검증.

    KMP_DUPLICATE_LIB_OK=TRUE python gen_check.py --data data/merged1.5M.csv.gz \
        --synth synth_diff.npz synth_gan.npz --out gen_check

점검 항목
  1) 주변분포(marginal): 열별 평균/표준편차가 실데이터와 비슷한가 + 히스토그램 비교
  2) 동역학 일관성: pos_err 는 다음 스텝에 -(속도)*dt 만큼 움직여야 한다.
     residual = ‖Δpos_err + v·dt‖ 는 실데이터에서 '목표점 이동량'(≈ speed·dt) 수준으로 작다.
     생성 전이가 물리를 못 배웠다면 이 값이 크게 튄다 → 학습에 넣어도 해로울 수 있음.
  3) 쿼터니언 노름(=1) 유지 여부
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gen_common import real_matrix, split_matrix, load_synth

DT = 1.0 / 100.0          # 제어 주파수 100Hz
PLOT_DIMS = [("pos_err x", 0), ("vel x", 7), ("angvel z", 12), ("action x", 16), ("reward", 35)]


def dyn_residual(s, ns):
    """‖Δpos_err + v·dt‖ — 작을수록 동역학적으로 말이 된다."""
    d_err = ns[:, 0:3] - s[:, 0:3]
    return np.linalg.norm(d_err + s[:, 7:10] * DT, axis=1)


def quat_norm(s):
    return np.linalg.norm(s[:, 3:7], axis=1)


def summarize(name, x):
    s, a, ns, r = split_matrix(x)
    res = dyn_residual(s, ns)
    print(f"\n[{name}]  n={len(x):,}")
    print(f"  reward      mean={r.mean():+.3f}  std={r.std():.3f}  min={r.min():+.3f} max={r.max():+.3f}")
    print(f"  action|max| ={np.abs(a).max():.3f}")
    print(f"  quat norm   mean={quat_norm(s).mean():.4f}  (1.0 이어야 정상)")
    print(f"  dyn residual mean={res.mean():.4f} m  median={np.median(res):.4f}  p95={np.percentile(res,95):.4f}")
    return res


def main():
    ap = argparse.ArgumentParser(description="합성 전이 품질 점검")
    ap.add_argument("--data", default="data/merged1.5M.csv.gz")
    ap.add_argument("--real_rows", type=int, default=300000, help="비교용 실데이터 표본 수")
    ap.add_argument("--synth", nargs="+", required=True, help="합성 npz 파일들")
    ap.add_argument("--out", default="gen_check")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    real, _, _ = real_matrix(args.data, args.real_rows)
    sets = [("real", real)]
    for p in args.synth:
        s, a, ns, r = load_synth(p)[:4]
        sets.append((os.path.basename(p).replace(".npz", ""),
                     np.concatenate([s, a, ns, r], axis=1).astype(np.float32)))

    residuals = {name: summarize(name, x) for name, x in sets}

    # 열별 평균/표준편차 차이 요약(실데이터 대비)
    print("\n[열별 통계 괴리(실데이터 대비, 표준편차 단위)]")
    r_mu, r_sd = real.mean(0), real.std(0) + 1e-6
    for name, x in sets[1:]:
        gap = np.abs(x.mean(0) - r_mu) / r_sd
        print(f"  {name:<12} mean gap: 평균={gap.mean():.3f} 최대={gap.max():.3f} "
              f"| std 비율 평균={np.mean((x.std(0)+1e-6)/r_sd):.3f}")

    # 그림: 주요 열 히스토그램 + 동역학 잔차
    n = len(PLOT_DIMS) + 1
    fig, axs = plt.subplots(1, n, figsize=(3.4 * n, 3.2))
    for ax, (label, d) in zip(axs, PLOT_DIMS):
        for name, x in sets:
            ax.hist(x[:, d], bins=80, density=True, histtype="step", lw=1.3, label=name)
        ax.set_title(label, fontsize=9); ax.tick_params(labelsize=7)
    ax = axs[-1]
    for name, res in residuals.items():
        ax.hist(np.clip(res, 0, np.percentile(residuals["real"], 99.9) * 5),
                bins=80, density=True, histtype="step", lw=1.3, label=name)
    ax.set_title("dynamics residual (m)", fontsize=9); ax.tick_params(labelsize=7)
    axs[0].legend(fontsize=7)
    fig.suptitle("real vs synthetic transitions", fontsize=11)
    fig.tight_layout()
    png = os.path.join(args.out, "compare.png")
    plt.savefig(png, dpi=120, bbox_inches="tight"); plt.close()
    print(f"\nplot -> {png}")


if __name__ == "__main__":
    main()
