"""Diffusion(DDPM) 데이터 증강 — 전이 36차원 벡터의 분포를 학습해 새 전이를 생성.

SynthER(Lu et al., 2023) 방식: 오프라인 RL 데이터셋의 (s,a,s',r) 전이를 통째로
생성모델로 학습하고, 생성한 합성 전이를 실제 데이터에 섞어 오프라인 RL을 재학습한다.

    python gen_diffusion.py --data data/merged1.5M.csv.gz --n 500000 --out synth_diff.npz

학습: 잡음예측(epsilon-prediction) DDPM, T=1000 선형 베타.
생성: DDIM(결정적, eta=0) 50스텝 — 품질 대비 CPU 시간이 20배 싸다.
"""
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn

from gen_common import DIM, real_matrix, fit_norm, postprocess, save_synth, MLP, batches


class Denoiser(nn.Module):
    """x_t 와 t 를 받아 더해진 잡음 eps 를 예측."""

    def __init__(self, dim=DIM, hidden=512, depth=3, t_dim=64):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.body = MLP(dim + t_dim, dim, hidden=hidden, depth=depth)

    def t_embed(self, t):
        """sinusoidal timestep embedding (Transformer/DDPM 표준)."""
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    def forward(self, x, t):
        return self.body(torch.cat([x, self.t_mlp(self.t_embed(t))], dim=1))


class DDPM:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        beta = torch.linspace(1e-4, 0.02, T, device=device)
        self.alpha_bar = torch.cumprod(1.0 - beta, dim=0)          # \bar{alpha}_t
        self.device = device

    def add_noise(self, x0, t, eps):
        ab = self.alpha_bar[t].unsqueeze(1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    @torch.no_grad()
    def ddim_sample(self, model, n, steps=50, batch=20000, log=True):
        """DDIM(eta=0) 역과정으로 n개 샘플 생성. 정규화 공간의 (n,DIM) 반환."""
        ts = torch.linspace(self.T - 1, 0, steps, device=self.device).long()
        out, done, t0 = [], 0, time.time()
        while done < n:
            b = min(batch, n - done)
            x = torch.randn(b, DIM, device=self.device)
            for i, t in enumerate(ts):
                tt = t.repeat(b)
                eps = model(x, tt)
                ab = self.alpha_bar[t]
                x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
                if i + 1 < len(ts):                                # 다음(더 낮은) 잡음 수준으로
                    ab_prev = self.alpha_bar[ts[i + 1]]
                    x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
                else:
                    x = x0
            out.append(x.cpu())
            done += b
            if log:
                el = time.time() - t0
                print(f"  sampled {done:,}/{n:,}  ({el:.0f}s, {done/max(el,1e-9):.0f}/s)", flush=True)
        return torch.cat(out)[:n]


def main():
    ap = argparse.ArgumentParser(description="Diffusion 전이 생성(SynthER 방식)")
    ap.add_argument("--data", default="data/merged1.5M.csv.gz")
    ap.add_argument("--max_rows", type=int, default=None,
                    help="실데이터 일부만으로 생성기 학습(저데이터 증강 실험). 예: 100000")
    ap.add_argument("--subset", choices=["even", "head"], default="even",
                    help="일부만 쓸 때 고르는 방식(even=전체에 흩어진 에피소드)")
    ap.add_argument("--shapes_keep", nargs="+", default=None,
                    help="이 도형만으로 생성기 학습(경로 일반화 실험). 빼놓은 도형이 합성을 통해"
                         " 새어 들어가지 않게 한다")
    ap.add_argument("--shape_labels", default="shape_labels.csv",
                    help="label_shapes.py 가 만든 에피소드-도형 매핑")
    ap.add_argument("--n", type=int, default=500_000, help="생성할 전이 수")
    ap.add_argument("--steps", type=int, default=30_000, help="학습 스텝")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=1000, help="diffusion 총 스텝")
    ap.add_argument("--sample_steps", type=int, default=50, help="DDIM 역과정 스텝")
    ap.add_argument("--sample_batch", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--delta", type=lambda v: str(v).lower() != "false", default=True,
                    help="s' 대신 변화량(s'-s)을 생성(기본 True) — 동역학 일관성에 결정적")
    ap.add_argument("--save_model", default=None, help="학습된 생성기 저장(재샘플링용)")
    ap.add_argument("--load_model", default=None, help="저장된 생성기로 학습 없이 샘플링만")
    ap.add_argument("--out", default="synth_diff.npz")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    print(f"=== Diffusion(DDPM) | device={dev} | train {args.steps:,} | gen {args.n:,} ===")

    if args.load_model:                       # 학습 건너뛰고 저장된 생성기로 샘플링만
        ck = torch.load(args.load_model, map_location=dev, weights_only=False)
        mu, sd, lo, hi, args.delta = ck["mu"], ck["sd"], ck["lo"], ck["hi"], ck["delta"]
        model = Denoiser(hidden=ck["hidden"], depth=ck["depth"]).to(dev)
        model.load_state_dict(ck["model"])
        dif = DDPM(T=ck["T"], device=dev)
        print(f"loaded generator <- {args.load_model}")
    else:
        x_raw, lo, hi = real_matrix(args.data, args.max_rows, delta=args.delta, subset=args.subset,
                                    shapes_keep=args.shapes_keep, shape_labels=args.shape_labels)
        mu, sd = fit_norm(x_raw)
        X = torch.as_tensor((x_raw - mu) / sd)

        model = Denoiser(hidden=args.hidden, depth=args.depth).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        dif = DDPM(T=args.T, device=dev)

        t0, it = time.time(), batches(len(X), args.batch_size)
        for step in range(1, args.steps + 1):
            x0 = X[next(it)].to(dev)
            t = torch.randint(0, args.T, (len(x0),), device=dev)
            eps = torch.randn_like(x0)
            loss = ((model(dif.add_noise(x0, t, eps), t) - eps) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % max(1, args.steps // 10) == 0:
                print(f"step {step:,}/{args.steps:,}  loss={loss.item():.4f}  "
                      f"({step/(time.time()-t0):.0f} it/s)", flush=True)

        if args.save_model:
            torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "lo": lo, "hi": hi,
                        "hidden": args.hidden, "depth": args.depth, "T": args.T,
                        "delta": args.delta}, args.save_model)
            print(f"[OK] generator -> {args.save_model}")

    print("sampling...", flush=True)
    model.eval()
    z = dif.ddim_sample(model, args.n, steps=args.sample_steps, batch=args.sample_batch).numpy()
    x = postprocess(z * sd + mu, lo, hi)                 # 원 단위 복원 + 범위 클리핑
    save_synth(args.out, x, delta=args.delta)


if __name__ == "__main__":
    main()
