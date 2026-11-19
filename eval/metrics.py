"""Precision, recall, density, coverage, per-digit metrics, and FID."""
import sys
import subprocess
import numpy as np
import torch


def _knn_radii(X, k, batch_size=512):
    """Compute distance to k-th nearest neighbor (excluding self) for points in X."""
    X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X.float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_dev = X.to(device)

    radii = []
    # k+1 because the point itself has distance 0
    k_target = min(k + 1, len(X))
    for i in range(0, len(X), batch_size):
        batch = X_dev[i:i + batch_size]
        d = torch.cdist(batch, X_dev)
        topk = d.topk(k_target, dim=1, largest=False)
        radii.append(topk.values[:, -1].cpu())

    return torch.cat(radii)


def knn_precision_recall(real_feats, fake_feats, k=5, batch_size=512):
    """k-NN precision and recall (Kynkäänniemi et al. 2019).
    real_feats: (N_real, D)
    fake_feats: (N_fake, D)
    """
    real = torch.tensor(real_feats, dtype=torch.float32) if not isinstance(real_feats, torch.Tensor) else real_feats.float()
    fake = torch.tensor(fake_feats, dtype=torch.float32) if not isinstance(fake_feats, torch.Tensor) else fake_feats.float()

    real_radii = _knn_radii(real, k=k, batch_size=batch_size)
    fake_radii = _knn_radii(fake, k=k, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_dev = real.to(device)
    fake_dev = fake.to(device)
    real_radii_dev = real_radii.to(device)
    fake_radii_dev = fake_radii.to(device)

    # Precision: share of fake points inside at least one real hypersphere
    precision_count = 0
    for i in range(0, len(fake), batch_size):
        fb = fake_dev[i:i + batch_size]
        # (batch, N_real)
        d = torch.cdist(fb, real_dev)
        inside = (d <= real_radii_dev.unsqueeze(0)).any(dim=1)
        precision_count += inside.sum().item()
    precision = precision_count / len(fake)

    # Recall: share of real points inside at least one fake hypersphere
    recall_count = 0
    for i in range(0, len(real), batch_size):
        rb = real_dev[i:i + batch_size]
        # (batch, N_fake)
        d = torch.cdist(rb, fake_dev)
        inside = (d <= fake_radii_dev.unsqueeze(0)).any(dim=1)
        recall_count += inside.sum().item()
    recall = recall_count / len(real)

    return {"precision": float(precision), "recall": float(recall)}


def density_coverage(real_feats, fake_feats, k=5, batch_size=512):
    """Density and coverage (Naeem et al. 2020).
    real_feats: (N_real, D)
    fake_feats: (N_fake, D)
    """
    real = torch.tensor(real_feats, dtype=torch.float32) if not isinstance(real_feats, torch.Tensor) else real_feats.float()
    fake = torch.tensor(fake_feats, dtype=torch.float32) if not isinstance(fake_feats, torch.Tensor) else fake_feats.float()

    real_radii = _knn_radii(real, k=k, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_dev = real.to(device)
    fake_dev = fake.to(device)
    real_radii_dev = real_radii.to(device)

    # Density: for each fake sample, count how many real balls contain it, divided by k
    density_total = 0.0
    for i in range(0, len(fake), batch_size):
        fb = fake_dev[i:i + batch_size]
        d = torch.cdist(fb, real_dev)
        inside_counts = (d <= real_radii_dev.unsqueeze(0)).float().sum(dim=1)
        density_total += (inside_counts / k).sum().item()
    density = density_total / len(fake)

    # Coverage: fraction of real samples whose ball contains at least one fake sample
    coverage_count = 0
    for i in range(0, len(real), batch_size):
        rb = real_dev[i:i + batch_size]
        d = torch.cdist(rb, fake_dev)
        radii_batch = real_radii_dev[i:i + batch_size].unsqueeze(1)
        has_fake = (d <= radii_batch).any(dim=1)
        coverage_count += has_fake.sum().item()
    coverage = coverage_count / len(real)

    return {"density": float(density), "coverage": float(coverage)}


def per_digit_recall(real_feats, fake_feats, real_labels, k=5):
    """Per-digit recall: k-NN recall restricted to real test images of digit d."""
    real_labels = np.array(real_labels)
    # fake radii computed from all fake samples
    fake = torch.tensor(fake_feats, dtype=torch.float32) if not isinstance(fake_feats, torch.Tensor) else fake_feats.float()
    fake_radii = _knn_radii(fake, k=k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_dev = fake.to(device)
    fake_radii_dev = fake_radii.to(device)

    results = {}
    for d in range(10):
        mask = (real_labels == d)
        if mask.sum() == 0:
            results[d] = 0.0
            continue
        real_d = torch.tensor(real_feats[mask], dtype=torch.float32).to(device)
        dists = torch.cdist(real_d, fake_dev)
        inside = (dists <= fake_radii_dev.unsqueeze(0)).any(dim=1)
        results[d] = float(inside.float().mean().item())

    return results


def per_digit_precision(fake_feats, fake_preds, real_feats, k=5):
    """Per-digit precision: precision of fake samples classified as digit d."""
    fake_preds = np.array(fake_preds)
    real = torch.tensor(real_feats, dtype=torch.float32) if not isinstance(real_feats, torch.Tensor) else real_feats.float()
    real_radii = _knn_radii(real, k=k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_dev = real.to(device)
    real_radii_dev = real_radii.to(device)

    results = {}
    for d in range(10):
        mask = (fake_preds == d)
        if mask.sum() == 0:
            results[d] = 0.0
            continue
        fake_d = torch.tensor(fake_feats[mask], dtype=torch.float32).to(device)
        dists = torch.cdist(fake_d, real_dev)
        inside = (dists <= real_radii_dev.unsqueeze(0)).any(dim=1)
        results[d] = float(inside.float().mean().item())

    return results


def fid_from_dirs(real_dir, fake_dir, dims=2048):
    """Compute FID using pytorch-fid."""
    try:
        from pytorch_fid import fid_score
        fid = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=50,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dims=dims
        )
        return float(fid)
    except Exception as e:
        # Fallback to CLI
        cmd = [sys.executable, "-m", "pytorch_fid", real_dir, fake_dir, "--dims", str(dims)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            lines = res.stdout.strip().split("\n")
            for line in lines:
                if "FID" in line or line.strip().replace(".", "").replace(" ", "").isdigit():
                    return float(line.strip().split(":")[-1].strip())
            return float(lines[-1].strip())
        raise RuntimeError(f"FID calculation failed: {e}\n{res.stderr}")
