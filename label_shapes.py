"""에피소드별 도형 라벨링 — 데이터셋에 도형 열이 없어서 궤적 기하로 역추정한다.

'삼각형·사각형만 학습 → 처음 보는 도형 평가' 실험을 하려면 어떤 에피소드가 어떤 도형인지
알아야 한다. lookahead 벡터(lx,ly,lz)는 경로의 진행 방향이므로, 그 방향이 어떻게 변하는지로
도형을 구분할 수 있다.

    python label_shapes.py --data data/merged1.5M_hard_v2.csv.gz --out shape_labels.csv
    python label_shapes.py --verbose        # 블록 경계를 눈으로 확인
"""
import argparse
import numpy as np
import pandas as pd


def path_stats(l):
    """직진 비율과 '직진 구간의 평균 길이'.

    코너 개수를 세는 방식은 외란(킥) 때문에 가짜 코너가 잡혀 못 쓴다. 대신 경로의 모양이
    직진 구간 길이에 남는 성질을 쓴다 — 둘레가 같을 때 변이 적은 도형일수록 한 변이 길다:
        삼각형(3변) > 사각형(4변) > 오각형(5변) >> 원(직진 구간 자체가 없음)
    킥은 짧은 구간만 흔들어서 이 통계에는 거의 영향을 주지 않는다.
    """
    u = l / (np.linalg.norm(l, axis=1, keepdims=True) + 1e-9)
    ang = np.degrees(np.arccos(np.clip((u[:-1] * u[1:]).sum(1), -1, 1)))
    if len(ang) < 10:
        return 0.0, 0.0
    straight = ang < 0.05
    runs, cur = [], 0
    for s in straight:
        if s:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    long_runs = [r for r in runs if r >= 20]      # 20스텝(0.2초) 이상만 '변'으로 인정
    return float(np.mean(straight)), (float(np.mean(long_runs)) if long_runs else 0.0)


def main():
    ap = argparse.ArgumentParser(description="에피소드 도형 라벨링")
    ap.add_argument("--data", default="data/merged1.5M_hard_v2.csv.gz")
    ap.add_argument("--out", default="shape_labels.csv")
    ap.add_argument("--verbose", action="store_true", help="블록 경계 확인용 상세 출력")
    args = ap.parse_args()

    df = pd.read_csv(args.data, usecols=["episode_id", "lx", "ly", "lz"])
    rows = []
    for eid, g in df.groupby("episode_id"):
        sf, rl = path_stats(g[["lx", "ly", "lz"]].to_numpy(np.float32))
        rows.append((int(eid), len(g), sf, rl))
    r = pd.DataFrame(rows, columns=["episode_id", "rows", "straight_frac", "run_len"])
    r = r.sort_values("episode_id").reset_index(drop=True)

    #### 수집 스크립트가 도형별로 순서대로 돌기 때문에 에피소드가 도형 블록으로 묶여 있다.
    if args.verbose:
        grp = r.index // 20
        print("[20 에피소드 묶음별 통계] — 값이 바뀌는 지점이 도형 경계")
        print(r.groupby(grp).agg(ep_from=("episode_id", "min"), ep_to=("episode_id", "max"),
                                 straight=("straight_frac", "mean"),
                                 run_len=("run_len", "mean")).round(3).to_string())

    #### 도형은 연속 블록이므로 에피소드별로 따로 판정하면 안 된다 — 킥을 많이 맞은 다각형
    #### 에피소드가 직진비율이 낮게 나와 '원'으로 오분류된다. 이동 중앙값으로 경계 한 곳만
    #### 찾아 그 앞을 통째로 원으로 본다.
    smooth = r["straight_frac"].rolling(11, center=True, min_periods=1).median()
    after = np.flatnonzero((smooth > 0.10).to_numpy())
    boundary = int(after[0]) if len(after) else len(r)
    r["shape"] = "circle"

    #### 나머지 다각형도 수집 순서대로 연속 블록이라 3등분한 뒤,
    #### 직진구간 길이 순서(오각형 < 사각형 < 삼각형)로 이름을 붙인다.
    poly = r.index[boundary:].to_numpy()
    if len(poly) >= 3:
        blocks = np.array_split(poly, 3)
        order = sorted(range(3), key=lambda i: r.loc[blocks[i], "run_len"].mean())
        for rank, bi in enumerate(order):
            r.loc[blocks[bi], "shape"] = ["pentagon", "square", "triangle"][rank]

    r.to_csv(args.out, index=False)
    print("\n[도형별 요약]")
    for s, g in r.groupby("shape"):
        e = g["episode_id"].to_numpy()
        print(f"  {s:<9} id {e.min():>3}~{e.max():<3} | {len(e):>3}개 | {g['rows'].sum():>9,} rows"
              f" | 직진 {g['straight_frac'].mean():.3f} | 변길이 {g['run_len'].mean():>6.1f}")
    print(f"\n[OK] {len(r)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
