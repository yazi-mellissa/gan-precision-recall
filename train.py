"""Train a GAN on MNIST with the fixed generator.

Final recipe (kept; local FID 8.85 / 7.03 over 3 seeds at 150 / 300 epochs, 6.22 at 500, 5.62 at 1,000 epochs):
    python train.py --loss bce --sn --beta1 0.5 --lr 2e-4 --lr_d 1e-4 --epochs 1000 --ema 0.999 --seed 0 \
        --out_dir runs/e15_bce_mlpsn_ema_1000ep_seed0
    -> runs/e15_bce_mlpsn_ema_1000ep_seed0/ema/G.pth is the generator we submit (checkpoints/G.pth).

The other options were tried and not kept (see README); they stay here so every run of experiments/ can be reproduced.
Outputs in --out_dir: config.json, log.jsonl (one line per epoch), G.pth, D.pth, ema/G.pth (with --ema).
"""
import os
import copy
import json
import time
import random
import argparse
import subprocess
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from model import Generator, Discriminator
from losses import bce_d, bce_g, hinge_d, hinge_g, chi2_d, prdiv_g, prdiv_g_bce


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def get_data_loader(batch_size, train=True):
    """MNIST scaled to [-1, 1] to match the generator's tanh. $DATA points to the cluster copy."""
    data_path = os.environ.get("DATA", "data")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    ds = datasets.MNIST(root=data_path, train=train, transform=transform, download=(data_path == "data"))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=0,
                                       pin_memory=torch.cuda.is_available())


def load_state(model, path, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})


def log_epoch(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def d_step_loss(args, D, x_real, x_fake):
    """Discriminator loss for the chosen --loss (prdiv uses a chi-squared critic, the others a BCE/hinge D)."""
    if args.loss in ("bce", "bce_prdiv"):
        return bce_d(D, x_real, x_fake, r1_gamma=args.r1_gamma)
    if args.loss == "hinge":
        return hinge_d(D, x_real, x_fake, r1_gamma=args.r1_gamma)
    return chi2_d(D, x_real, x_fake, r1_gamma=args.r1_gamma, t_clip=args.t_clip)


def g_step_loss(args, D, x_fake):
    if args.loss == "bce":
        return bce_g(D, x_fake)
    if args.loss == "hinge":
        return hinge_g(D, x_fake)
    if args.loss == "bce_prdiv":
        return prdiv_g_bce(D, x_fake, lam=args.lam)
    return prdiv_g(D, x_fake, lam=args.lam, t_clip=args.t_clip, soft_tau=args.soft_tau)


def train_epoch(args, G, D, G_ema, loader, G_opt, D_opt, device):
    """One pass over MNIST: a D step on every batch, a G step every n_critic batches, then the EMA update."""
    G.train()
    D.train()
    sums, n = {}, 0
    for step, (x, _) in enumerate(loader):
        x = x.view(-1, 784).to(device)
        bs = x.shape[0]

        d_loss, d_stats = d_step_loss(args, D, x, G(torch.randn(bs, 100, device=device)).detach())
        D_opt.zero_grad()
        d_loss.backward()
        D_opt.step()

        if step % args.n_critic != 0:
            continue
        g_loss, g_stats = g_step_loss(args, D, G(torch.randn(bs, 100, device=device)))
        G_opt.zero_grad()
        g_loss.backward()
        G_opt.step()

        # EMA of G's weights (Yazici et al. 2019): averages out the oscillation of GAN training
        if G_ema is not None:
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.lerp_(p, 1.0 - args.ema)

        for k, v in {**d_stats, **g_stats, "d_loss": d_loss.item(), "g_loss": g_loss.item()}.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    return {k: v / n for k, v in sums.items()}


def train_d_only(args, G, D, loader, D_opt, device, epochs, tag, log_path):
    """Train D alone on a frozen G: used for the D warm-up before a fine-tune and the D refresh before OBRS."""
    G.eval()
    for ep in range(1, epochs + 1):
        D.train()
        t0, sums, n = time.time(), {}, 0
        for x, _ in loader:
            x = x.view(-1, 784).to(device)
            with torch.no_grad():
                x_fake = G(torch.randn(x.shape[0], 100, device=device))
            d_loss, d_stats = d_step_loss(args, D, x, x_fake)
            D_opt.zero_grad()
            d_loss.backward()
            D_opt.step()
            for k, v in {**d_stats, "d_loss": d_loss.item()}.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        avg = {k: v / n for k, v in sums.items()}
        avg.update({"epoch": f"{tag}_{ep}", "epoch_time": time.time() - t0})
        log_epoch(log_path, avg)
        print(f"  {tag} {ep}: {avg}")


def main():
    parser = argparse.ArgumentParser(description="Train a GAN on MNIST (fixed generator).")
    # Kept: the final recipe uses these
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "hinge", "prdiv", "bce_prdiv"],
                        help="bce is kept; hinge, prdiv and bce_prdiv were tried.")
    parser.add_argument("--sn", action="store_true", help="Spectral normalisation on every D layer (kept).")
    parser.add_argument("--ema", type=float, default=0.0, help="EMA decay of G's weights, e.g. 0.999 (kept). Saved as ema/G.pth.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate of G.")
    parser.add_argument("--lr_d", type=float, default=None, help="Learning rate of D (two time-scale update). Defaults to --lr.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1 (0.5 in every run since E2).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--snapshots", type=int, nargs="*", default=[],
                        help="With --ema: also keep the EMA generator of these epochs in ema_ep<N>/G.pth (FID vs training length).")
    # Tried, not kept
    parser.add_argument("--r1_gamma", type=float, default=0.0, help="R1 penalty on D, e.g. 10 (E8 base model, replaced by --sn).")
    parser.add_argument("--d_arch", type=str, default="mlp", choices=["mlp", "conv"], help="conv: lower recall.")
    parser.add_argument("--n_critic", type=int, default=1, help="D steps per G step.")
    parser.add_argument("--lr_decay", action="store_true", help="Linear decay of both learning rates to 0.")
    parser.add_argument("--lam", type=float, default=1.0, help="prdiv: lambda of the PR-divergence.")
    parser.add_argument("--t_clip", type=float, default=None, help="prdiv: clamp the critic T to [-2, t_clip].")
    parser.add_argument("--soft_tau", type=float, default=0.0, help="prdiv: smooth max in f_lambda (0 = hard max).")
    parser.add_argument("--init_from", type=str, default=None, help="Fine-tune: start from G.pth/D.pth of this run.")
    parser.add_argument("--d_warmup_epochs", type=int, default=0, help="Fine-tune: train D alone first.")
    parser.add_argument("--d_refresh_epochs", type=int, default=0, help="After training, retrain D alone (saved as D_refresh.pth, for OBRS).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    config = {**vars(args), "device": str(device), "git_hash": get_git_hash(),
              "start_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    config_path = os.path.join(args.out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    loader = get_data_loader(args.batch_size)
    G = Generator(g_output_dim=784).to(device)
    D = Discriminator(784, arch=args.d_arch, sn=args.sn).to(device)
    if args.init_from:
        print(f"Loading from {args.init_from}")
        load_state(G, os.path.join(args.init_from, "G.pth"), device)
        load_state(D, os.path.join(args.init_from, "D.pth"), device)
    G_ema = copy.deepcopy(G).eval() if args.ema > 0 else None

    lr_d = args.lr_d if args.lr_d is not None else args.lr
    G_opt = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    D_opt = optim.Adam(D.parameters(), lr=lr_d, betas=(args.beta1, 0.999))
    schedulers = []
    if args.lr_decay:
        decay = lambda e: 1.0 - e / args.epochs
        schedulers = [optim.lr_scheduler.LambdaLR(G_opt, decay), optim.lr_scheduler.LambdaLR(D_opt, decay)]

    log_path = os.path.join(args.out_dir, "log.jsonl")
    if args.init_from and args.d_warmup_epochs > 0:
        train_d_only(args, G, D, loader, D_opt, device, args.d_warmup_epochs, "warmup", log_path)

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        metrics = train_epoch(args, G, D, G_ema, loader, G_opt, D_opt, device)
        for sch in schedulers:
            sch.step()
        metrics.update({"epoch": epoch, "epoch_time": time.time() - t0})
        log_epoch(log_path, metrics)

        if epoch % 10 == 0 or epoch == args.epochs:  # checkpoint every 10 epochs
            torch.save(G.state_dict(), os.path.join(args.out_dir, "G.pth"))
            torch.save(D.state_dict(), os.path.join(args.out_dir, "D.pth"))
            if G_ema is not None:
                os.makedirs(os.path.join(args.out_dir, "ema"), exist_ok=True)
                torch.save(G_ema.state_dict(), os.path.join(args.out_dir, "ema", "G.pth"))
            print(f"  Epoch {epoch}: {metrics}")
        if G_ema is not None and epoch in args.snapshots:
            os.makedirs(os.path.join(args.out_dir, f"ema_ep{epoch}"), exist_ok=True)
            torch.save(G_ema.state_dict(), os.path.join(args.out_dir, f"ema_ep{epoch}", "G.pth"))
    print(f"Training done in {time.time() - t_start:.1f}s")

    if args.d_refresh_epochs > 0:
        D_refresh_opt = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        train_d_only(args, G, D, loader, D_refresh_opt, device, args.d_refresh_epochs, "refresh", log_path)
        torch.save(D.state_dict(), os.path.join(args.out_dir, "D_refresh.pth"))

    config.update({"end_time": time.strftime("%Y-%m-%d %H:%M:%S"), "total_time": time.time() - t_start})
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
