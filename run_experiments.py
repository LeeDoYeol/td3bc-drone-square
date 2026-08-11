"""증강 비교 실험 7설정을 한 번에 실행 (학습 → 체크포인트 best 선택 → 결과표).

총 전이 수를 1.5M 으로 고정하고 실제/합성 비율만 바꿔, '증강이 실제 데이터를 얼마나
대체·보강하는가' 를 한 번에 비교한다.

    python run_experiments.py --data data/merged1.5M_hard_v2.csv.gz \
        --diff synth_diff_hv2.npz --gan synth_gan_hv2.npz --att_d_gain 1.0 --device cuda

각 설정마다 runs_<tag>/ (모델·체크포인트) 와 select_<tag>/ (궤적 그림) 이 생기고,
마지막에 results.md / results.csv 로 비교표를 남긴다. 중간에 끊겨도 이미 끝난 설정은
건너뛰므로 그대로 다시 실행하면 이어서 진행된다.
"""
import os
import re
import csv
import sys
import time
import argparse
import subprocess


def configs(diff, gan):
    """(태그, train.py 추가인자, 설명) — 총량은 모두 1.5M."""
    return [
        ("c1_real15",       ["--real_rows", "1500000"],
         "원본 1.5M (기준선)"),
        ("c2_real10_dif05", ["--real_rows", "1000000", "--extra_data", diff, "--extra_rows", "500000"],
         "원본 1.0M + diffusion 0.5M"),
        ("c3_real05_dif10", ["--real_rows", "500000", "--extra_data", diff, "--extra_rows", "1000000"],
         "원본 0.5M + diffusion 1.0M"),
        ("c4_real10_gan05", ["--real_rows", "1000000", "--extra_data", gan, "--extra_rows", "500000"],
         "원본 1.0M + GAN 0.5M"),
        ("c5_real05_gan10", ["--real_rows", "500000", "--extra_data", gan, "--extra_rows", "1000000"],
         "원본 0.5M + GAN 1.0M"),
        ("c6_dif15",        ["--real_rows", "0", "--extra_data", diff, "--extra_rows", "1500000"],
         "diffusion 1.5M (합성만)"),
        ("c7_gan15",        ["--real_rows", "0", "--extra_data", gan, "--extra_rows", "1500000"],
         "GAN 1.5M (합성만)"),
    ]


def run(cmd, log_path):
    """서브프로세스 실행 — 화면과 로그 파일에 동시에 남긴다."""
    print("$ " + " ".join(cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line)
        p.wait()
    if p.returncode != 0:
        raise SystemExit(f"[FAIL] {' '.join(cmd)} (exit {p.returncode})")


def main():
    ap = argparse.ArgumentParser(description="증강 비교 7설정 실행")
    ap.add_argument("--data", default="data/merged1.5M_hard_v2.csv.gz")
    ap.add_argument("--diff", default="synth_diff_hv2.npz", help="diffusion 합성 npz")
    ap.add_argument("--gan", default="synth_gan_hv2.npz", help="GAN 합성 npz")
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=2.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--shapes", nargs="+", default=["line", "triangle", "square", "circle", "star"])
    ap.add_argument("--seed", type=int, default=500, help="평가용 held-out 시드")
    ap.add_argument("--att_d_gain", type=float, default=1.0,
                    help="평가 시 자세 D게인 — 데이터 수집 때와 같게 (hard_v2=1.0, merged1.5M=0.3)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="+", default=None, help="특정 태그만 실행")
    ap.add_argument("--logs", default="logs")
    args = ap.parse_args()

    os.makedirs(args.logs, exist_ok=True)
    py = sys.executable
    rows, t_all = [], time.time()

    for tag, extra, desc in configs(args.diff, args.gan):
        if args.only and tag not in args.only:
            continue
        run_dir, sel_dir = f"runs_{tag}", f"select_{tag}"
        t0 = time.time()
        print(f"\n{'='*70}\n[{tag}] {desc}\n{'='*70}", flush=True)

        if not os.path.exists(os.path.join(run_dir, "td3bc_model.pt")):
            run([py, "train.py", "--data", args.data, "--steps", str(args.steps),
                 "--save_every", str(args.save_every), "--alpha", str(args.alpha),
                 "--reward_norm", "False", "--grad_clip", str(args.grad_clip),
                 "--device", args.device, "--out", run_dir] + extra,
                os.path.join(args.logs, f"{tag}_train.log"))
        else:
            print(f"[skip] {run_dir} 이미 학습됨", flush=True)

        sel_log = os.path.join(args.logs, f"{tag}_select.log")
        if not os.path.exists(os.path.join(sel_dir, "best_trajectories.png")):
            run([py, "select_best.py", "--run_dir", run_dir, "--shapes", *args.shapes,
                 "--seed", str(args.seed), "--att_d_gain", str(args.att_d_gain),
                 "--out", sel_dir], sel_log)
        else:
            print(f"[skip] {sel_dir} 이미 평가됨", flush=True)

        # select_best 로그에서 best 결과 회수
        text = open(sel_log, encoding="utf-8").read() if os.path.exists(sel_log) else ""
        m = re.search(r"\[BEST\]\s+(\S+)\s+mean_err=([\d.]+)", text)
        best_ckpt, mean_err = (m.group(1), float(m.group(2))) if m else ("?", float("nan"))
        per = {}
        for line in text.splitlines():                      # 체크포인트별 표에서 best 행의 도형별 값
            if m and line.startswith(best_ckpt):
                vals = line.split()[1:]
                if len(vals) == 1 + len(args.shapes):
                    per = dict(zip(args.shapes, [float(v) for v in vals[1:]]))
        rows.append(dict(tag=tag, desc=desc, mean_err=mean_err, best_ckpt=best_ckpt,
                         minutes=round((time.time() - t0) / 60, 1), **per))
        print(f"[{tag}] mean_err={mean_err:.4f} m  ({rows[-1]['minutes']} min)", flush=True)

    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r["mean_err"])
    cols = ["tag", "desc", "mean_err", "best_ckpt", "minutes"] + args.shapes
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)
    with open("results.md", "w", encoding="utf-8") as f:
        f.write(f"# 증강 비교 결과 ({args.data}, {args.steps:,} steps)\n\n")
        f.write("| 설정 | 5도형 평균 | best ckpt | " + " | ".join(args.shapes) + " |\n")
        f.write("|---|---|---|" + "---|" * len(args.shapes) + "\n")
        for r in rows_sorted:
            shp = " | ".join(f"{r.get(s, float('nan')):.2f}" for s in args.shapes)
            f.write(f"| {r['desc']} | **{r['mean_err']:.3f} m** | {r['best_ckpt']} | {shp} |\n")
        f.write(f"\n총 소요: {(time.time()-t_all)/3600:.1f} 시간\n")

    print("\n" + "=" * 70)
    for r in rows_sorted:
        print(f"  {r['mean_err']:.4f} m   {r['desc']}")
    print(f"\n[OK] results.md / results.csv 저장  (총 {(time.time()-t_all)/3600:.1f}시간)")
    print("산출물 압축: python collect_outputs.py")


if __name__ == "__main__":
    main()
