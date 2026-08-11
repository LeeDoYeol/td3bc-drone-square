"""합성 전이 npz → 배포용 CSV(.gz) 변환 (동료 공유용).

    python export_synth.py --synth synth_diff_1.5M.npz --out dist/synth_diffusion_1.5M.csv.gz

원본 데이터셋 CSV 는 '연속된 에피소드'라 next_state 를 다음 행에서 얻지만, 합성 전이는
서로 독립적인 (s,a,s',r) 하나하나다. 그래서 다음 상태를 다음 행에 두지 않고 n_* 열로
같은 행에 명시한다 — 이렇게 해야 순서에 의존하지 않고 어떤 오프라인 RL 코드에서도
그대로 쓸 수 있다.

열(36+2): tx-x ... lz (상태 16) | ax,ay,az (행동 3) | reward | done | n_tx-x ... n_lz (다음상태 16)
"""
import os
import argparse
import numpy as np
import pandas as pd

from data import S_COLS, A_COLS
from gen_common import load_synth


def main():
    ap = argparse.ArgumentParser(description="합성 전이 npz → CSV(.gz)")
    ap.add_argument("--synth", required=True, help="gen_diffusion.py / gen_gan.py 산출 npz")
    ap.add_argument("--out", required=True, help="출력 CSV 경로(.gz 권장)")
    args = ap.parse_args()

    s, a, ns, r, nd = load_synth(args.synth)
    df = pd.DataFrame(s, columns=S_COLS)
    for i, c in enumerate(A_COLS):
        df[c] = a[:, i]
    df["reward"] = r[:, 0]
    df["done"] = (nd[:, 0] < 0.5)
    for i, c in enumerate(S_COLS):
        df[f"n_{c}"] = ns[:, i]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.6g",
              compression="gzip" if args.out.endswith(".gz") else None)
    mb = os.path.getsize(args.out) / 1e6
    print(f"[OK] {len(df):,} transitions x {len(df.columns)} cols -> {args.out} ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
