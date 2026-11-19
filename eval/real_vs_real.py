"""Real vs Real baseline: MNIST train subset against MNIST test set."""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.features import load_pngs, vgg_features, get_pr_indices, export_real_pngs
from eval.metrics import knn_precision_recall, density_coverage


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=9984)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--vgg_layer", type=str, default="fc2_relu")
    parser.add_argument("--out", type=str, default="eval/assets/real_vs_real.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Running Real vs Real evaluation on {device}...")

    # Ensure real test PNGs exist
    real_test_dir = "eval/assets/real_test_png"
    if not os.path.exists(real_test_dir) or len(os.listdir(real_test_dir)) < 10000:
        export_real_pngs(real_test_dir, n=10000)

    # Export real train PNGs
    real_train_dir = "eval/assets/real_train_png"
    if not os.path.exists(real_train_dir) or len(os.listdir(real_train_dir)) < args.n:
        data_path = os.environ.get("DATA", "data")
        download = data_path == "data"
        train_ds = torchvision.datasets.MNIST(root=data_path, train=True, download=download,
                                              transform=transforms.ToTensor())
        os.makedirs(real_train_dir, exist_ok=True)
        rng = np.random.RandomState(1234)
        indices = rng.permutation(len(train_ds))[:args.n]
        for idx, orig_i in enumerate(sorted(indices)):
            img, _ = train_ds[orig_i]
            torchvision.utils.save_image(img, os.path.join(real_train_dir, f"{idx}.png"))
        print(f"Exported {args.n} train images to {real_train_dir}")

    pr_indices = get_pr_indices(args.n)
    test_imgs = load_pngs(real_test_dir, indices=pr_indices)
    train_imgs = load_pngs(real_train_dir, n=args.n)

    print("Extracting VGG features...")
    test_vgg = vgg_features(test_imgs, device=device, layer=args.vgg_layer)
    train_vgg = vgg_features(train_imgs, device=device, layer=args.vgg_layer)

    print("Computing metrics (test vs train)...")
    pr = knn_precision_recall(test_vgg, train_vgg, k=args.k)
    dc = density_coverage(test_vgg, train_vgg, k=args.k)

    results = {
        "n": args.n,
        "k": args.k,
        "vgg_layer": args.vgg_layer,
        "precision": pr["precision"],
        "recall": pr["recall"],
        "density": dc["density"],
        "coverage": dc["coverage"],
    }

    print(f"\nReal vs Real Ceiling (VGG k={args.k}):")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"Density:   {results['density']:.4f}")
    print(f"Coverage:  {results['coverage']:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
