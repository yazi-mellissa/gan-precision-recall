"""Full evaluation CLI: compute VGG P/R, density/coverage, FID, CNN metrics, per-digit metrics."""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.features import load_pngs, vgg_features, get_pr_indices, cache_features, export_real_pngs
from eval.metrics import (
    knn_precision_recall,
    density_coverage,
    per_digit_recall,
    per_digit_precision,
    fid_from_dirs
)
from eval.classifier import load_cnn, classify_images, digit_stats


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def evaluate(samples_dir, real_dir="eval/assets/real_test_png", cache_dir="eval/cache",
             vgg_layer="fc2_relu", k=5, n_pr=9984, device=None, skip_fid=False,
             run_dir=None):
    if device is None:
        device = get_device()

    if not os.path.exists(real_dir) or len(os.listdir(real_dir)) < 10000:
        print(f"Exporting real test PNGs to {real_dir}...")
        export_real_pngs(real_dir, n=10000)

    results = {
        "samples_dir": samples_dir,
        "real_dir": real_dir,
        "vgg_layer": vgg_layer,
        "k": k,
        "n_pr": n_pr,
        "run_dir": run_dir,
    }

    # Load sampler.json if present
    sampler_cfg = None
    possible_sampler_paths = [
        os.path.join(samples_dir, "sampler.json"),
        os.path.join(run_dir, "sampler.json") if run_dir else None,
        "checkpoints/sampler.json"
    ]
    for sp in possible_sampler_paths:
        if sp and os.path.exists(sp):
            try:
                with open(sp) as f:
                    sampler_cfg = json.load(f)
                break
            except Exception:
                pass
    results["sampler_config"] = sampler_cfg

    t0 = time.time()
    pr_indices = get_pr_indices(n_pr)

    print("Loading PNG images (PNG round trip)...")
    real_imgs = load_pngs(real_dir, indices=pr_indices)
    fake_imgs = load_pngs(samples_dir, n=n_pr)

    results["n_real"] = len(real_imgs)
    results["n_fake"] = len(fake_imgs)
    print(f"Loaded {len(real_imgs)} real test images and {len(fake_imgs)} fake images.")

    # VGG Feature extraction
    print(f"Extracting VGG features ({vgg_layer})...")
    real_cache = os.path.join(cache_dir, f"real_vgg_{vgg_layer}_{n_pr}.npy")
    real_vgg = cache_features(real_imgs, real_cache, vgg_features, device=device, layer=vgg_layer)
    fake_vgg = vgg_features(fake_imgs, device=device, layer=vgg_layer)

    # VGG Precision and Recall
    print("Computing k-NN precision and recall (VGG)...")
    pr = knn_precision_recall(real_vgg, fake_vgg, k=k)
    results["vgg_precision"] = pr["precision"]
    results["vgg_recall"] = pr["recall"]

    # VGG Density and Coverage
    print("Computing density and coverage (VGG)...")
    dc = density_coverage(real_vgg, fake_vgg, k=k)
    results["vgg_density"] = dc["density"]
    results["vgg_coverage"] = dc["coverage"]

    # FID
    if not skip_fid:
        print("Computing FID (Inception pool3)...")
        try:
            fid = fid_from_dirs(real_dir, samples_dir, dims=2048)
            results["fid"] = fid
            print(f"FID: {fid:.4f}")
        except Exception as e:
            print(f"FID failed: {e}")
            results["fid"] = None
    else:
        results["fid"] = None

    # CNN classifier evaluation
    print("Evaluating with MNIST CNN...")
    try:
        cnn = load_cnn(device=device)
        fake_all = load_pngs(samples_dir)
        preds, confs, fake_cnn_feats = classify_images(fake_all, cnn, device)
        ds = digit_stats(preds, confs)
        results["cnn"] = ds

        # Penultimate layer P/R
        _, _, real_cnn_feats = classify_images(real_imgs, cnn, device)
        cnn_pr = knn_precision_recall(real_cnn_feats.numpy(), fake_cnn_feats[:n_pr].numpy(), k=k)
        results["cnn_precision"] = cnn_pr["precision"]
        results["cnn_recall"] = cnn_pr["recall"]

        # Per-digit recall using ground-truth test labels
        import torchvision
        from torchvision import transforms
        data_path = os.environ.get("DATA", "data")
        download = data_path == "data"
        test_ds = torchvision.datasets.MNIST(root=data_path, train=False, download=download,
                                             transform=transforms.ToTensor())
        real_labels = [test_ds[i][1] for i in pr_indices]
        pdr = per_digit_recall(real_vgg, fake_vgg, real_labels, k=k)
        results["per_digit_recall"] = pdr

        # Per-digit precision
        pdp = per_digit_precision(fake_vgg[:n_pr], preds[:n_pr], real_vgg, k=k)
        results["per_digit_precision"] = pdp
    except Exception as e:
        print(f"CNN evaluation error: {e}")
        results["cnn_error"] = str(e)

    results["eval_time_seconds"] = float(time.time() - t0)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated samples against MNIST test set.")
    parser.add_argument("--samples_dir", type=str, default="samples")
    parser.add_argument("--real_dir", type=str, default="eval/assets/real_test_png")
    parser.add_argument("--cache_dir", type=str, default="eval/cache")
    parser.add_argument("--vgg_layer", type=str, default="fc2_relu", choices=["fc2_relu", "fc2"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n_pr", type=int, default=9984)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(args.samples_dir, "results.json")

    results = evaluate(
        samples_dir=args.samples_dir,
        real_dir=args.real_dir,
        cache_dir=args.cache_dir,
        vgg_layer=args.vgg_layer,
        k=args.k,
        n_pr=args.n_pr,
        skip_fid=args.skip_fid,
        run_dir=args.run_dir
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete. Results written to {args.out}")
    print(f"VGG P: {results.get('vgg_precision', 'N/A'):.4f} | R: {results.get('vgg_recall', 'N/A'):.4f}")
    if results.get("fid") is not None:
        print(f"FID: {results['fid']:.4f}")


if __name__ == "__main__":
    main()
