"""GAN(WGAN-GP) 데이터 증강 — diffusion 과 같은 36차원 전이를 생성(비교군).

    python gen_gan.py --data data/merged1.5M.csv.gz --n 500000 --out synth_gan.npz

WGAN-GP(Gulrajani et al., 2017): 판별기를 1-Lipschitz 로 제약하는 gradient penalty 를
써서 표형(tabular) 데이터에서도 모드 붕괴 없이 비교적 안정적으로 학습된다.
생성은 forward 1회라 diffusion 보다 훨씬 빠르다(품질은 이 실험에서 비교 대상).
"""
import time
import argparse
import numpy as np
import torch

from gen_common import DIM, real_matrix, fit_norm, postprocess, save_synth, MLP, batches


def grad_penalty(critic, real, fake, device):
    """판별기 기울기 노름이 1에서 벗어난 정도(WGAN-GP 핵심 항)."""
    a = torch.rand(len(real), 1, device=device)
    mid = (a * real + (1 - a) * fake).requires_grad_(True)
    d = critic(mid)
    g = torch.autograd.grad(d, mid, torch.ones_like(d), create_graph=True)[0]
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


def main():
    ap = argparse.ArgumentParser(description="WGAN-GP 전이 생성")
    ap.add_argument("--data", default="data/merged1.5M.csv.gz")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--n", type=int, default=500_000, help="생성할 전이 수")
    ap.add_argument("--steps", type=int, default=15_000, help="생성기 업데이트 수")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_critic", type=int, default=5, help="생성기 1회당 판별기 업데이트 수")
    ap.add_argument("--gp", type=float, default=10.0, help="gradient penalty 계수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--delta", type=lambda v: str(v).lower() != "false", default=True,
                    help="s' 대신 변화량(s'-s)을 생성(기본 True) — 동역학 일관성에 결정적")
    ap.add_argument("--save_model", default=None, help="학습된 생성기 저장(재샘플링용)")
    ap.add_argument("--load_model", default=None, help="저장된 생성기로 학습 없이 샘플링만")
    ap.add_argument("--out", default="synth_gan.npz")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    print(f"=== WGAN-GP | device={dev} | train {args.steps:,} | gen {args.n:,} ===")

    if args.load_model:                       # 학습 건너뛰고 저장된 생성기로 샘플링만
        ck = torch.load(args.load_model, map_location=dev, weights_only=False)
        mu, sd, lo, hi, args.delta = ck["mu"], ck["sd"], ck["lo"], ck["hi"], ck["delta"]
        args.latent = ck["latent"]
        G = MLP(args.latent, DIM, hidden=ck["hidden"], depth=ck["depth"]).to(dev)
        G.load_state_dict(ck["G"])
        print(f"loaded generator <- {args.load_model}")
    else:
        x_raw, lo, hi = real_matrix(args.data, args.max_rows, delta=args.delta)
        mu, sd = fit_norm(x_raw)
        X = torch.as_tensor((x_raw - mu) / sd)

        G = MLP(args.latent, DIM, hidden=args.hidden, depth=args.depth).to(dev)
        D = MLP(DIM, 1, hidden=args.hidden, depth=args.depth, act=torch.nn.LeakyReLU).to(dev)
        optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
        optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

        t0, it = time.time(), batches(len(X), args.batch_size)
        for step in range(1, args.steps + 1):
            for _ in range(args.n_critic):                 # 판별기(critic) 업데이트
                real = X[next(it)].to(dev)
                fake = G(torch.randn(len(real), args.latent, device=dev)).detach()
                lossD = D(fake).mean() - D(real).mean() + args.gp * grad_penalty(D, real, fake, dev)
                optD.zero_grad(set_to_none=True); lossD.backward(); optD.step()

            fake = G(torch.randn(args.batch_size, args.latent, device=dev))   # 생성기 업데이트
            lossG = -D(fake).mean()
            optG.zero_grad(set_to_none=True); lossG.backward(); optG.step()

            if step % max(1, args.steps // 10) == 0:
                print(f"step {step:,}/{args.steps:,}  D={lossD.item():.4f}  G={lossG.item():.4f}  "
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)

        if args.save_model:
            torch.save({"G": G.state_dict(), "mu": mu, "sd": sd, "lo": lo, "hi": hi,
                        "latent": args.latent, "hidden": args.hidden, "depth": args.depth,
                        "delta": args.delta}, args.save_model)
            print(f"[OK] generator -> {args.save_model}")

    print("sampling...", flush=True)
    G.eval()
    out, done = [], 0
    with torch.no_grad():
        while done < args.n:
            b = min(50000, args.n - done)
            out.append(G(torch.randn(b, args.latent, device=dev)).cpu())
            done += b
    z = torch.cat(out)[:args.n].numpy()
    x = postprocess(z * sd + mu, lo, hi)
    save_synth(args.out, x, delta=args.delta)


if __name__ == "__main__":
    main()
