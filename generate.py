"""Platform entry point: load checkpoints/G.pth and write 10,000 PNGs to samples/0.png ... samples/9999.png.

We submit plain sampling, z ~ N(0, I): truncation and rejection sampling gave no real gain on the
final (EMA) generator, see README. The tried samplers are in tried/sampling.py.
"""
import os
import time
import argparse
import torch
import torchvision

from model import Generator


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        torch.set_num_threads(4)  # the platform runs on CPU
        return torch.device("cpu")


def load_G(ckpt_dir, device):
    G = Generator(g_output_dim=784).to(device)
    sd = torch.load(os.path.join(ckpt_dir, "G.pth"), map_location=device, weights_only=True)
    G.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    G.eval()
    return G


@torch.no_grad()
def generate(G, n, batch_size, device, out_dir, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    i = 0
    while i < n:
        bs = min(batch_size, n - i)
        x = G(torch.randn(bs, 100, device=device)).cpu()
        x = (x.view(bs, 1, 28, 28) + 1) / 2  # tanh output [-1, 1] -> pixels [0, 1]
        for img in x:
            torchvision.utils.save_image(img, os.path.join(out_dir, f"{i}.png"))
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 10,000 MNIST samples.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Folder containing G.pth")
    parser.add_argument("--out_dir", type=str, default="samples")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    t0 = time.time()
    G = load_G(args.ckpt_dir, device)
    generate(G, args.n_samples, args.batch_size, device, args.out_dir, seed=args.seed)
    print(f"Generated {args.n_samples} images in {time.time() - t0:.1f}s")
