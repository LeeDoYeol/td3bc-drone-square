"""실험 B — 단일 도형 학습 → 전체 도형 평가 (경로 일반화).

도형 하나만 0.1M 학습하고 5도형에서 평가한다. 학습 도형 외에는 전부 '처음 보는 경로'이므로,
증강이 데이터에 없는 경로 유형을 만들어 낼 수 있는지 직접 시험할 수 있다.

    python run_shape_exp.py --device cuda                    # 생성 + 학습 + 평가 전부
    python run_shape_exp.py --only b3_circle b4_circle_dif    # 일부만
    python run_shape_exp.py --no_eval                         # 학습까지만(pybullet 없는 머신)

각 설정의 생성기는 그 설정이 쓰는 실제 데이터(도형·크기)만 보고 학습한다. 전체로 학습한
생성기를 쓰면 빼놓은 도형이 합성을 통해 새어 들어가 실험이 무의미해진다.
"""
import os
import re
import csv
import sys
import json
import time
import argparse
import subprocess

#### (태그, 학습 도형, 증강 사용 여부) — 도형 None 이면 4도형 균등(mixed)
CONFIGS = [
    ("b1_square",       ["square"],   False),
    ("b2_square_dif",   ["square"],   True),
    ("b3_circle",       ["circle"],   False),
    ("b4_circle_dif",   ["circle"],   True),
    ("b5_triangle",     ["triangle"], False),
    ("b6_triangle_dif", ["triangle"], True),
    ("b7_mixed",        None,         False),
    ("b8_mixed_dif",    None,         True),
]


def sh(cmd, log_path=None):
    print("\n$ " + " ".join(cmd), flush=True)
    f = open(log_path, "w", encoding="utf-8") if log_path else None
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        if f:
            f.write(line)
    p.wait()
    if f:
        f.close()
    if p.returncode != 0:
        raise SystemExit(f"[FAIL] {' '.join(cmd)} (exit {p.returncode})")


def gen_name(shapes):
    return "synth_b_" + ("mixed" if shapes is None else "_".join(shapes)) + ".npz"


def main():
    ap = argparse.ArgumentParser(description="실험 B — 단일 도형 학습")
    ap.add_argument("--data", default="data/merged1.5M_hard_v2.csv.gz")
    ap.add_argument("--labels", default="shape_labels_hv2.csv")
    ap.add_argument("--real_rows", type=int, default=100000, help="학습에 쓸 실제 전이 수")
    ap.add_argument("--synth_rows", type=int, default=1400000, help="증강 설정의 합성 전이 수")
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gen_steps", type=int, default=30000, help="생성기 학습 스텝")
    ap.add_argument("--shapes", nargs="+",
                    default=["line", "triangle", "square", "circle", "star"], help="평가 도형")
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--att_d_gain", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--no_eval", action="store_true", help="학습까지만(pybullet 없는 머신용)")
    ap.add_argument("--logs", default="logs")
    args = ap.parse_args()

    os.makedirs(args.logs, exist_ok=True)
    py = sys.executable
    rows, t_all = [], time.time()

    for tag, shapes, use_syn in CONFIGS:
        if args.only and tag not in args.only:
            continue
        run_dir, sel_dir = f"runs_{tag}", f"select_{tag}"
        t0 = time.time()
        label = "4도형 균등" if shapes is None else "+".join(shapes)
        print(f"\n{'='*70}\n[{tag}] 학습 도형: {label}"
              f"{' + diffusion 증강' if use_syn else ''}\n{'='*70}", flush=True)

        extra = ["--real_rows", str(args.real_rows)]
        if shapes:
            extra += ["--shapes_keep", *shapes, "--shape_labels", args.labels]

        #### 증강 설정이면 그 도형·그 크기만 본 생성기를 먼저 만든다
        if use_syn:
            syn = gen_name(shapes)
            if os.path.exists(syn):
                print(f"[skip] {syn} 이미 있음", flush=True)
            else:
                gcmd = [py, "gen_diffusion.py", "--data", args.data,
                        "--max_rows", str(args.real_rows), "--subset", "even",
                        "--n", str(args.synth_rows), "--steps", str(args.gen_steps),
                        "--save_model", syn.replace(".npz", ".pt"), "--out", syn,
                        "--device", args.device]
                if shapes:
                    gcmd += ["--shapes_keep", *shapes, "--shape_labels", args.labels]
                sh(gcmd, os.path.join(args.logs, f"{tag}_gen.log"))
            extra += ["--extra_data", syn, "--extra_rows", str(args.synth_rows)]

        #### 설정이 바뀌면 다시 학습한다(예전 모델 조용히 재사용 방지)
        meta_path = os.path.join(run_dir, "run_meta.json")
        sig = {"data": args.data, "steps": args.steps, "alpha": args.alpha, "extra": extra}
        old = None
        if os.path.exists(meta_path):
            try:
                old = json.load(open(meta_path, encoding="utf-8"))
            except Exception:
                old = None
        if os.path.exists(os.path.join(run_dir, "td3bc_model.pt")) and old == sig:
            print(f"[skip] {run_dir} 같은 설정으로 이미 학습됨", flush=True)
        else:
            sh([py, "train.py", "--data", args.data, "--steps", str(args.steps),
                "--save_every", str(args.save_every), "--alpha", str(args.alpha),
                "--reward_norm", "False", "--grad_clip", "1.0",
                "--device", args.device, "--out", run_dir] + extra,
               os.path.join(args.logs, f"{tag}_train.log"))
            json.dump(sig, open(meta_path, "w", encoding="utf-8"))

        if args.no_eval:
            print(f"[{tag}] 학습 완료 ({(time.time()-t0)/60:.1f}분) — 평가는 다른 머신에서",
                  flush=True)
            continue

        sel_log = os.path.join(args.logs, f"{tag}_select.log")
        sh([py, "select_best.py", "--run_dir", run_dir, "--shapes", *args.shapes,
            "--seed", str(args.seed), "--att_d_gain", str(args.att_d_gain),
            "--stride", str(args.stride), "--out", sel_dir], sel_log)

        text = open(sel_log, encoding="utf-8").read()
        m = re.search(r"\[BEST\]\s+(\S+)\s+mean_err=([\d.]+)\s*m(?:\s+coverage=([\d.]+)%)?", text)
        best, err = (m.group(1), float(m.group(2))) if m else ("?", float("nan"))
        cov = float(m.group(3)) if (m and m.group(3)) else float("nan")
        per = {}
        for line in text.splitlines():
            if m and line.startswith(best):
                v = line.replace("%", "").split()[1:]
                if len(v) == 2 + len(args.shapes):
                    per = dict(zip(args.shapes, [float(x) for x in v[2:]]))
        rows.append(dict(tag=tag, trained_on=label, aug=use_syn, coverage=cov, mean_err=err,
                         best_ckpt=best, minutes=round((time.time() - t0) / 60, 1), **per))
        print(f"[{tag}] 완주율 {cov:.0f}%  오차 {err:.4f} m  ({rows[-1]['minutes']}분)", flush=True)

    if not rows:
        return
    cols = ["tag", "trained_on", "aug", "coverage", "mean_err", "best_ckpt", "minutes"] + args.shapes
    with open("results_shape.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\n" + "=" * 70)
    for r in rows:
        print(f"  {r['trained_on']:<12} {'+증강' if r['aug'] else '원본만':<6} "
              f"완주 {r['coverage']:>3.0f}%  오차 {r['mean_err']:.4f}")
    print(f"\n[OK] results_shape.csv  (총 {(time.time()-t_all)/3600:.1f}시간)")


if __name__ == "__main__":
    main()
