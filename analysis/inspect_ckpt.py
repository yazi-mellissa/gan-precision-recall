"""Compare a checkpoint's weight statistics with a fresh Generator init."""
import os
import sys
import torch
import numpy as np
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import Generator


def weight_stats(state_dict):
    stats = {}
    for name, param in state_dict.items():
        p = param.float().numpy()
        stats[name] = {"mean": float(np.mean(p)), "std": float(np.std(p)),
                       "min": float(np.min(p)), "max": float(np.max(p)),
                       "shape": list(p.shape)}
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/G.pth")
    parser.add_argument("--grid_out", default="tools/ckpt_grid.png")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Fresh init
    G_fresh = Generator(784)
    fresh_sd = G_fresh.state_dict()
    fresh_stats = weight_stats(fresh_sd)

    # Checkpoint
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    ckpt_sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    ckpt_sd = {k.replace("module.", ""): v for k, v in ckpt_sd.items()}
    ckpt_stats = weight_stats(ckpt_sd)

    print("=" * 70)
    print(f"{'Layer':<20} {'Fresh mean':>10} {'Fresh std':>10} {'Ckpt mean':>10} {'Ckpt std':>10} {'Match?':>8}")
    print("=" * 70)

    is_trained = False
    for name in fresh_stats:
        f = fresh_stats[name]
        c = ckpt_stats.get(name, {"mean": float("nan"), "std": float("nan")})
        match = abs(f["std"] - c["std"]) < 0.01 and abs(f["mean"] - c["mean"]) < 0.01
        flag = "YES" if match else "NO"
        if not match:
            is_trained = True
        print(f"{name:<20} {f['mean']:>10.5f} {f['std']:>10.5f} {c['mean']:>10.5f} {c['std']:>10.5f} {flag:>8}")

    print("=" * 70)
    if is_trained:
        print("VERDICT: Checkpoint appears TRAINED (weights differ from fresh init)")
    else:
        print("VERDICT: Checkpoint appears UNTRAINED (weights match fresh init)")

    # Generate 8x8 sample grid
    G = Generator(784).to(device)
    G.load_state_dict(ckpt_sd)
    G.eval()

    torch.manual_seed(42)
    with torch.no_grad():
        z = torch.randn(64, 100, device=device)
        x = G(z)
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = x.reshape(64, 1, 28, 28)

    import torchvision
    os.makedirs(os.path.dirname(args.grid_out) or ".", exist_ok=True)
    torchvision.utils.save_image(x, args.grid_out, nrow=8, padding=2)
    print(f"Sample grid saved to {args.grid_out}")


if __name__ == "__main__":
    main()
