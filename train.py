"""TD3+BC 오프라인 학습 (동료 제공 merged 데이터셋: square+triangle+circle).

    python train.py --data data/merged.csv --steps 1000000 --device cuda
    python train.py --data data/merged_1.csv --max_rows 40000 --steps 5000   # 스모크
    python train.py --obs_noise 0.1                                          # 후처리 관측노이즈(선택)

관측 s=16차원, 행동 a=3차원. 학습 후 model(td3bc_model.pt)과 정규화(norm.npz) 저장.
"""
import os
import time
import argparse
import numpy as np
import torch

from data import load_dataset, normalize_states, ReplayBuffer, STATE_DIM, ACTION_DIM
from td3bc import TD3_BC


def resolve_device(device):
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser(description="TD3+BC 오프라인 학습")
    ap.add_argument("--data", default="data/merged1.5M.csv.gz", help="CSV(.gz 가능) 파일/폴더")
    ap.add_argument("--steps", type=int, default=1_000_000, help="총 gradient 스텝")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_rows", type=int, default=None, help="데이터 서브샘플(스모크)")
    ap.add_argument("--obs_noise", type=float, default=0.0,
                    help="후처리 관측 노이즈 std(정규화 상태 기준, 예 0.1). 0이면 미사용")
    ap.add_argument("--reward_norm", type=lambda v: str(v).lower() != "false", default=True,
                    help="보상을 std로 나눠 정규화(기본 True, critic 안정화)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--log_every", type=int, default=5000)
    # TD3+BC 하이퍼파라미터 (sfujim 기본값)
    ap.add_argument("--alpha", type=float, default=2.5)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--policy_freq", type=int, default=2)
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="critic/actor gradient norm clip (0=off) — critic 폭발 억제")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="learning rate (critic/actor) — 낮추면 critic 폭발 완화(느린 학습)")
    ap.add_argument("--extra_data", default=None,
                    help="합성 전이 npz(쉼표로 여러 개) — gen_diffusion.py / gen_gan.py 산출물")
    ap.add_argument("--real_rows", type=int, default=None,
                    help="실데이터를 이 개수로 잘라 사용(0이면 합성만). 증강 비율 실험용")
    ap.add_argument("--extra_rows", type=int, default=None,
                    help="합성 데이터를 이 개수로 잘라 사용(파일당)")
    ap.add_argument("--save_every", type=int, default=0,
                    help=">0이면 N스텝마다 체크포인트 저장(ckpt_{step}.pt) → 나중에 best 선택")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"=== TD3+BC | device={device} | steps={args.steps:,} ===")

    # 데이터
    state, action, next_state, reward, not_done, max_action = load_dataset(args.data, args.max_rows)
    if args.real_rows is not None:           # 0 이면 합성 데이터만으로 학습(대체 실험)
        state, action, next_state, reward, not_done = (
            x[:args.real_rows] for x in (state, action, next_state, reward, not_done))
        print(f"real rows -> {len(state):,}")
    # 합성 전이(diffusion/GAN) 를 실데이터에 이어 붙인다. 정규화는 아래에서 합쳐진 전체로 다시 계산.
    if args.extra_data:
        from gen_common import load_synth
        for path in args.extra_data.split(","):
            s2, a2, ns2, r2, nd2 = load_synth(path.strip())
            if args.extra_rows is not None:
                s2, a2, ns2, r2, nd2 = (x[:args.extra_rows] for x in (s2, a2, ns2, r2, nd2))
            state = np.concatenate([state, s2]); action = np.concatenate([action, a2])
            next_state = np.concatenate([next_state, ns2]); reward = np.concatenate([reward, r2])
            not_done = np.concatenate([not_done, nd2])
            print(f"+ synthetic {len(s2):,} <- {path.strip()}")
        max_action = float(np.abs(action).max())
    state, next_state, mean, std = normalize_states(state, next_state)
    print(f"transitions={len(state):,}  state_dim={state.shape[1]}  action_dim={action.shape[1]}")
    print(f"action range=[{action.min():.3f}, {action.max():.3f}] (max_action={max_action:.3f})  "
          f"reward mean={reward.mean():.4f}")
    # 보상 정규화 (스케일이 크면 critic 발산 → std로 나눠 안정화)
    if args.reward_norm:
        rstd = float(reward.std()) + 1e-6
        reward = reward / rstd
        print(f"reward normalized by std={rstd:.3f}")
    if args.obs_noise > 0:
        print(f"obs_noise={args.obs_noise} (후처리 관측 증강)")
    buffer = ReplayBuffer(state, action, next_state, reward, not_done, device, obs_noise=args.obs_noise)

    # 에이전트 (max_action 은 데이터에서 관측된 최대 행동 크기)
    agent = TD3_BC(STATE_DIM, ACTION_DIM, max_action=max_action, device=device,
                   discount=args.discount, tau=args.tau, policy_freq=args.policy_freq,
                   alpha=args.alpha, grad_clip=args.grad_clip, lr=args.lr)

    # 정규화 값 먼저 저장(모든 체크포인트가 공유; best 선택 때 필요)
    np.savez(os.path.join(args.out, "norm.npz"), mean=mean, std=std, max_action=max_action)
    if args.save_every:
        os.makedirs(os.path.join(args.out, "ckpts"), exist_ok=True)

    # 학습
    t0 = time.time()
    for t in range(1, args.steps + 1):
        info = agent.train(buffer, args.batch_size)
        if t % args.log_every == 0:
            el = time.time() - t0
            msg = f"step {t:,}/{args.steps:,}  critic={info['critic_loss']:.4f}"
            if "actor_loss" in info:
                msg += f"  actor={info['actor_loss']:.4f}  bc={info['bc_loss']:.4f}  Q={info['Q']:.3f}"
            msg += f"  ({t/el:.0f} it/s)"
            print(msg, flush=True)
        if args.save_every and t % args.save_every == 0:
            agent.save(os.path.join(args.out, "ckpts", f"ckpt_{t:07d}.pt"))

    # 최종 저장
    agent.save(os.path.join(args.out, "td3bc_model.pt"))
    print(f"[OK] model -> {os.path.join(args.out, 'td3bc_model.pt')}")
    print(f"[OK] norm  -> {os.path.join(args.out, 'norm.npz')}")


if __name__ == "__main__":
    main()
