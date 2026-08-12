"""전체 파이프라인 한 번에 실행 — 데이터 준비 → 증강 생성 → 품질점검 → 13설정 비교 → 산출물 압축.

    python run_all.py --device cuda

이미 만들어진 결과물이 있으면 그 단계는 건너뛴다. 중간에 끊겨도 같은 명령을 다시 넣으면
이어서 진행되므로, 오래 걸리는 실행을 나눠서 돌려도 된다.

주요 옵션
    --device cuda|cpu     학습·생성 장치
    --steps 300000        TD3+BC 학습 스텝 (설정 13개 각각)
    --skip_small          저데이터(0.1M) 실험 8·9번 제외 → 7설정만
    --only c1_real15 ...  특정 설정만
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


def sh(cmd, log_path=None):
    """서브프로세스 실행 — 화면 출력 + (선택) 로그 파일."""
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


def fetch_data(path=DATA_PATH):
    if os.path.exists(path) and os.path.getsize(path) > 10_000_000:
        print(f"[skip] 데이터셋 이미 있음: {path} "
              f"({os.path.getsize(path)/1e6:.0f} MB)")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"데이터셋 내려받는 중 -> {path}", flush=True)

    def hook(blocks, bs, total):
        if total > 0 and blocks % 200 == 0:
            print(f"  {blocks*bs/1e6:.0f} / {total/1e6:.0f} MB", flush=True)

    urllib.request.urlretrieve(DATA_URL, path, hook)
    print(f"[OK] {os.path.getsize(path)/1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser(description="전체 파이프라인 한 번에 실행")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=300000, help="TD3+BC 학습 스텝")
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--att_d_gain", type=float, default=1.0,
                    help="평가 시 자세 D게인 (hard_v2=1.0, merged1.5M=0.3)")
    ap.add_argument("--n_synth", type=int, default=1500000, help="전체 데이터 기반 합성 개수")
    ap.add_argument("--small_rows", type=int, default=100000, help="저데이터 실험의 원본 크기")
    ap.add_argument("--n_synth_small", type=int, default=1400000, help="저데이터 기반 합성 개수")
    ap.add_argument("--diff_steps", type=int, default=30000, help="diffusion 학습 스텝")
    ap.add_argument("--gan_steps", type=int, default=6000, help="GAN 생성기 업데이트 수")
    ap.add_argument("--alpha", type=float, default=2.5,
                    help="TD3+BC의 RL 비중. 높으면 보상 최대화, 낮으면 모방(BC) 위주")
    ap.add_argument("--stride", type=int, default=1,
                    help="체크포인트를 N개마다 하나씩만 평가(전체 시간의 대부분이 평가라 크게 단축)")
    ap.add_argument("--skip_small", action="store_true", help="저데이터 실험(8·9번) 제외")
    ap.add_argument("--only", nargs="+", default=None, help="특정 설정만 학습")
    ap.add_argument("--name", default="hard_v2", help="산출물 zip 이름 꼬리표")
    args = ap.parse_args()

    py = sys.executable
    os.makedirs("logs", exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print(f"전체 파이프라인 시작  device={args.device}  steps={args.steps:,}")
    print("=" * 70)

    # 1) 데이터셋
    fetch_data(args.data)

    # 2) 증강 데이터 생성 (전체 기반 + 저데이터 기반)
    jobs = [
        ("synth_diff_hv2.npz", [py, "gen_diffusion.py", "--data", args.data,
                                "--n", str(args.n_synth), "--steps", str(args.diff_steps),
                                "--save_model", "gen_diff_hv2.pt", "--out", "synth_diff_hv2.npz",
                                "--device", args.device]),
        ("synth_gan_hv2.npz", [py, "gen_gan.py", "--data", args.data,
                               "--n", str(args.n_synth), "--steps", str(args.gan_steps),
                               "--hidden", "384", "--save_model", "gen_gan_hv2.pt",
                               "--out", "synth_gan_hv2.npz", "--device", args.device]),
    ]
    if not args.skip_small:
        # 저데이터 실험용 생성기는 반드시 같은 0.1M 만 보고 학습해야 한다
        # (전체로 학습한 생성기를 쓰면 없다고 가정한 데이터가 새어 들어간다)
        jobs += [
            ("synth_diff_small.npz", [py, "gen_diffusion.py", "--data", args.data,
                                      "--max_rows", str(args.small_rows),
                                      "--n", str(args.n_synth_small), "--steps", str(args.diff_steps),
                                      "--save_model", "gen_diff_small.pt",
                                      "--out", "synth_diff_small.npz", "--device", args.device]),
            ("synth_gan_small.npz", [py, "gen_gan.py", "--data", args.data,
                                     "--max_rows", str(args.small_rows),
                                     "--n", str(args.n_synth_small), "--steps", str(args.gan_steps),
                                     "--hidden", "384", "--save_model", "gen_gan_small.pt",
                                     "--out", "synth_gan_small.npz", "--device", args.device]),
        ]
    for out, cmd in jobs:
        if os.path.exists(out):
            print(f"[skip] {out} 이미 있음")
        else:
            sh(cmd, os.path.join("logs", os.path.basename(out).replace(".npz", "_gen.log")))

    # 3) 합성 데이터 품질 점검 (학습 전에 물리 일관성 확인)
    synths = [o for o, _ in jobs]
    sh([py, "gen_check.py", "--data", args.data, "--synth", *synths, "--out", "gen_check_hv2"],
       os.path.join("logs", "gen_check.log"))

    # 4) 설정별 학습 + 체크포인트 best 선택
    cmd = [py, "run_experiments.py", "--data", args.data,
           "--diff", "synth_diff_hv2.npz", "--gan", "synth_gan_hv2.npz",
           "--steps", str(args.steps), "--save_every", str(args.save_every),
           "--alpha", str(args.alpha), "--stride", str(args.stride),
           "--att_d_gain", str(args.att_d_gain), "--device", args.device]
    if not args.skip_small:
        cmd += ["--diff_small", "synth_diff_small.npz", "--gan_small", "synth_gan_small.npz"]
    if args.only:
        cmd += ["--only", *args.only]
    sh(cmd)

    # 5) 산출물 압축
    sh([py, "collect_outputs.py", "--name", args.name])
    print(f"\n전체 소요: {(time.time()-t0)/3600:.1f} 시간")


if __name__ == "__main__":
    main()
