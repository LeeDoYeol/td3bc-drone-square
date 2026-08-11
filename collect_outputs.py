"""실험 산출물을 한 폴더로 모아 zip 으로 압축 — 전달용.

    python collect_outputs.py                       # artifacts_YYYYMMDD.zip 생성
    python collect_outputs.py --name hard_v2        # artifacts_hard_v2.zip

담는 것 (설정별):
    models/<tag>/best_model.pt   선택된 최고 체크포인트 (바로 평가/배포 가능)
    models/<tag>/norm.npz        관측 정규화 통계 (모델과 항상 같이 있어야 함)
    figures/<tag>.png            5도형 궤적 그림
    logs/                        학습·선택 로그 (체크포인트별 오차표 포함)
    results.md, results.csv      비교표
    generators/                  학습된 diffusion/GAN 생성기 (재생성용)
    gen_check/                   합성 데이터 품질 점검 그림

용량이 큰 것(체크포인트 전체 ckpts/, 합성 npz, 원본 데이터셋)은 제외한다.
"""
import os
import glob
import json
import shutil
import argparse
from datetime import datetime


def copy_if(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="산출물 수집 → zip")
    ap.add_argument("--name", default=datetime.now().strftime("%Y%m%d"), help="zip 이름 꼬리표")
    ap.add_argument("--runs_glob", default="runs_*", help="수집할 run 폴더 패턴")
    ap.add_argument("--keep_dir", action="store_true", help="압축 후 폴더를 지우지 않음")
    args = ap.parse_args()

    stage = f"artifacts_{args.name}"
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)

    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "configs": []}

    for run_dir in sorted(glob.glob(args.runs_glob)):
        if not os.path.isdir(run_dir):
            continue
        tag = run_dir[len("runs_"):]
        # best_model 이 없으면 최종 모델이라도 담는다
        got = copy_if(os.path.join(run_dir, "best_model.pt"), f"{stage}/models/{tag}/best_model.pt")
        if not got:
            got = copy_if(os.path.join(run_dir, "td3bc_model.pt"),
                          f"{stage}/models/{tag}/td3bc_model.pt")
        copy_if(os.path.join(run_dir, "norm.npz"), f"{stage}/models/{tag}/norm.npz")
        fig = copy_if(os.path.join(f"select_{tag}", "best_trajectories.png"),
                      f"{stage}/figures/{tag}.png")
        if got or fig:
            manifest["configs"].append({"tag": tag, "model": got, "figure": fig})

    for pat, dst in [("logs/*.log", "logs"), ("gen_check*/*.png", "gen_check"),
                     ("gen_*.pt", "generators")]:
        for f in glob.glob(pat):
            copy_if(f, os.path.join(stage, dst, os.path.basename(f)))

    for f in ("results.md", "results.csv"):
        copy_if(f, os.path.join(stage, f))

    with open(os.path.join(stage, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    zip_path = shutil.make_archive(stage, "zip", stage)
    if not args.keep_dir:
        shutil.rmtree(stage)

    size = os.path.getsize(zip_path) / 1e6
    print("\n" + "=" * 60)
    print(f"[OK] 산출물 {len(manifest['configs'])}개 설정 압축 완료 ({size:.1f} MB)")
    print(f"\n  {os.path.abspath(zip_path)}\n")
    print("=" * 60)
    print("이 파일 하나만 전달하면 됩니다 (모델 + 그림 + 비교표 + 로그 포함).")


if __name__ == "__main__":
    main()
