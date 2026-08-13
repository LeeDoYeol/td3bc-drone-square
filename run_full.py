"""실험 A + B 전체를 한 번에 — 데이터 준비, 생성기 6개, 설정 학습·평가, 산출물 압축.

    python run_full.py --device cuda

이미 만들어진 결과물이 있으면 그 단계를 건너뛴다. 중간에 끊겨도 같은 명령을 다시 넣으면
이어서 진행된다.

생성기를 크기별로 따로 만드는 이유: '원본 0.5M + 합성' 설정에서 전체 1.5M 을 본 생성기를
쓰면, 없다고 가정한 1.0M 이 합성 데이터를 통해 새어 들어와 저데이터 실험이 무의미해진다.
실험 B(단일 도형)도 같은 이유로 도형별 생성기를 쓴다 — run_shape_exp.py 가 알아서 만든다.
"""
import os
import sys
import time
import argparse
import subprocess
import urllib.request

DATA_URL = ("https://media.githubusercontent.com/media/subsubli/drone_simulation/main/"
            "gym_pybullet_drones/gym_pybullet_drones/examples/data_hard_v2/"
            "merged1.5M_hard_v2.csv.gz")
DATA_PATH = "data/merged1.5M_hard_v2.csv.gz"
LABELS = "shape_labels_hv2.csv"


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


def fetch_data():
    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 200_000_000:
        print(f"[skip] 데이터셋 있음 ({os.path.getsize(DATA_PATH)/1e6:.0f} MB)")
        return
    os.makedirs("data", exist_ok=True)
    print("데이터셋 내려받는 중...", flush=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"[OK] {os.path.getsize(DATA_PATH)/1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser(description="실험 A + B 전체 실행")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gen_steps", type=int, default=30000)
    ap.add_argument("--gen_frac", type=float, default=1.0,
                    help="생성할 합성 전이 수를 이 비율로 축소(파이프라인 점검용). 1.0=정상")
    ap.add_argument("--skip_a", action="store_true", help="실험 A 건너뜀")
    ap.add_argument("--skip_b", action="store_true", help="실험 B 건너뜀")
    ap.add_argument("--only_a", nargs="+", default=None)
    ap.add_argument("--only_b", nargs="+", default=None)
    ap.add_argument("--no_eval", action="store_true", help="학습까지만(pybullet 없는 머신)")
    ap.add_argument("--name", default="hard_v2")
    args = ap.parse_args()

    py = sys.executable
    os.makedirs("logs", exist_ok=True)
    t0 = time.time()

    fetch_data()

    if not os.path.exists(LABELS):
        sh([py, "label_shapes.py", "--data", DATA_PATH, "--out", LABELS],
           os.path.join("logs", "label.log"))
    else:
        print(f"[skip] {LABELS} 있음")

    # 실험 A 용 생성기 — 전체 / 0.5M / 0.1M 각각 그 데이터만 보고 학습
    if not args.skip_a:
        gens = [
            ("synth_diff_full.npz", "gen_diffusion.py", None, 1500000),
            ("synth_gan_full.npz",  "gen_gan.py",       None, 1500000),
            ("synth_diff_mid.npz",  "gen_diffusion.py", 500000, 1000000),
            ("synth_gan_mid.npz",   "gen_gan.py",       500000, 1000000),
            ("synth_diff_small.npz", "gen_diffusion.py", 100000, 1400000),
            ("synth_gan_small.npz",  "gen_gan.py",       100000, 1400000),
        ]
        for out, script, max_rows, n in gens:
            if os.path.exists(out):
                print(f"[skip] {out} 있음")
                continue
            cmd = [py, script, "--data", DATA_PATH, "--n", str(max(1000, int(n * args.gen_frac))),
                   "--steps", str(args.gen_steps if script.endswith("diffusion.py") else 6000),
                   "--save_model", out.replace(".npz", ".pt"), "--out", out,
                   "--device", args.device]
            if script.endswith("gan.py"):
                cmd += ["--hidden", "384"]
            if max_rows:
                cmd += ["--max_rows", str(max_rows), "--subset", "even"]
            sh(cmd, os.path.join("logs", out.replace(".npz", "_gen.log")))

        sh([py, "gen_check.py", "--data", DATA_PATH, "--synth",
            "synth_diff_full.npz", "synth_gan_full.npz",
            "synth_diff_small.npz", "synth_gan_small.npz", "--out", "gen_check"],
           os.path.join("logs", "gen_check.log"))

        cmd = [py, "run_experiments.py", "--data", DATA_PATH,
               "--diff", "synth_diff_full.npz", "--gan", "synth_gan_full.npz",
               "--diff_mid", "synth_diff_mid.npz", "--gan_mid", "synth_gan_mid.npz",
               "--diff_small", "synth_diff_small.npz", "--gan_small", "synth_gan_small.npz",
               "--steps", str(args.steps), "--save_every", str(args.save_every),
               "--alpha", str(args.alpha), "--stride", str(args.stride),
               "--att_d_gain", "1.0", "--device", args.device]
        if args.only_a:
            cmd += ["--only", *args.only_a]
        sh(cmd)

    # 실험 B — 단일 도형 학습(생성기는 run_shape_exp.py 가 도형별로 만든다)
    if not args.skip_b:
        cmd = [py, "run_shape_exp.py", "--data", DATA_PATH, "--labels", LABELS,
               "--steps", str(args.steps), "--save_every", str(args.save_every),
               "--alpha", str(args.alpha), "--stride", str(args.stride),
               "--gen_steps", str(args.gen_steps), "--gen_frac", str(args.gen_frac),
               "--device", args.device]
        if args.only_b:
            cmd += ["--only", *args.only_b]
        if args.no_eval:
            cmd += ["--no_eval"]
        sh(cmd)

    sh([py, "collect_outputs.py", "--name", args.name])
    print(f"\n전체 소요: {(time.time()-t0)/3600:.1f} 시간")


if __name__ == "__main__":
    main()
