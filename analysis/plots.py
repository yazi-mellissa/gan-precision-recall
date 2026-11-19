"""Generate figures and summary tables from evaluation JSONs."""
import os
import sys
import json
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def find_all_results(base_dir="runs"):
    """Find all results.json files in runs/ and sweeps/."""
    results = []
    # Direct run results
    patterns = [
        os.path.join(base_dir, "*", "results.json"),
        os.path.join(base_dir, "*", "sweeps", "*", "results.json"),
        "samples/results.json",
    ]
    seen = set()
    for pat in patterns:
        for p in glob.glob(pat):
            norm = os.path.normpath(p)
            if norm not in seen:
                seen.add(norm)
                try:
                    with open(p) as f:
                        data = json.load(f)
                    data["_file_path"] = norm
                    results.append(data)
                except Exception as e:
                    print(f"Error reading {p}: {e}")
    return results


def plot_pr_plane(results, out_path="figures/pr_plane.pdf"):
    """Produce the precision-recall plane figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Reference points
    ax.scatter([0.192], [0.467], color="crimson", marker="*", s=220, zorder=10,
               label="Platform VanillaGAN (R=0.192, P=0.467)")
    ax.scatter([0.700], [0.861], color="gold", edgecolors="black", marker="*", s=250, zorder=10,
               label="Platform BestOf 2025 (R=0.700, P=0.861)")

    # Group results
    categories = {
        "e1_vanilla": {"x": [], "y": [], "color": "darkgray", "marker": "o", "label": "E1 Vanilla"},
        "e2_baseline": {"x": [], "y": [], "color": "steelblue", "marker": "s", "label": "E2 Hygiene Baseline"},
        "prdiv": {"x": [], "y": [], "color": "forestgreen", "marker": "^", "label": "PR-divergence (Training)"},
        "obrs": {"x": [], "y": [], "color": "darkorange", "marker": "D", "label": "OBRS Rejection (Inference)"},
        "truncation": {"x": [], "y": [], "color": "purple", "marker": "x", "label": "Truncation (Reference)"},
        "other": {"x": [], "y": [], "color": "gray", "marker": ".", "label": "Other Runs"},
    }

    for r in results:
        p = r.get("vgg_precision")
        rec = r.get("vgg_recall")
        if p is None or rec is None:
            continue
        path = r.get("_file_path", "")
        sampler_cfg = r.get("sampler_config") or {}
        sampler_name = sampler_cfg.get("sampler", "")

        if sampler_name == "obrs" or "obrs" in path:
            cat = "obrs"
        elif sampler_name == "truncation" or "trunc" in path:
            cat = "truncation"
        elif "e1_" in path:
            cat = "e1_vanilla"
        elif "e2_" in path:
            cat = "e2_baseline"
        elif "prdiv" in path:
            cat = "prdiv"
        else:
            cat = "other"

        categories[cat]["x"].append(rec)
        categories[cat]["y"].append(p)

    for cat_key, cat_data in categories.items():
        if len(cat_data["x"]) > 0:
            ax.scatter(cat_data["x"], cat_data["y"],
                       c=cat_data["color"], marker=cat_data["marker"],
                       alpha=0.8, s=60, label=cat_data["label"])

    ax.set_xlabel("Recall (VGG k=5)", fontsize=12)
    ax.set_ylabel("Precision (VGG k=5)", fontsize=12)
    ax.set_title("Precision-Recall Trade-off on MNIST (VGG-16, k=5)", fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved P-R plane plot to {out_path}")


def plot_per_digit(results, out_path="figures/per_digit.pdf"):
    """Produce per-digit recall and confident digit histogram figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    digits = list(range(10))
    bar_width = 0.25
    x = np.arange(len(digits))

    # Pick representative runs (e.g. baseline, prdiv, obrs)
    selected = []
    for r in results:
        path = r.get("_file_path", "")
        if "per_digit_recall" in r and "cnn" in r and "digit_hist" in r["cnn"]:
            name = os.path.basename(os.path.dirname(path))
            selected.append((name, r))
            if len(selected) >= 3:
                break

    if not selected:
        print("No results with per_digit_recall / digit_hist found for per-digit plot.")
        plt.close()
        return

    for idx, (label, data) in enumerate(selected):
        pdr = data["per_digit_recall"]
        recalls = [pdr.get(str(d), pdr.get(d, 0.0)) for d in digits]
        hist = data["cnn"]["digit_hist"]
        offset = (idx - len(selected) / 2 + 0.5) * bar_width

        ax1.bar(x + offset, recalls, width=bar_width, label=label, alpha=0.85)
        ax2.bar(x + offset, hist, width=bar_width, label=label, alpha=0.85)

    ax1.set_ylabel("Per-digit Recall", fontsize=11)
    ax1.set_title("Per-Digit Recall (VGG k=5)", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2.set_xlabel("Digit Class", fontsize=11)
    ax2.set_ylabel("Confident Sample Count", fontsize=11)
    ax2.set_title("Digit Distribution (CNN conf >= 0.9)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in digits])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved per-digit analysis plot to {out_path}")


def plot_obrs_accept_reject(run_dir="runs/e2_baseline_seed0", out_path="figures/obrs_accept_reject.png", n_eval=1000):
    """Generate 8x8 grids of highest-r and lowest-r samples and r vs confidence scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torchvision
    from model import Generator, Discriminator
    from eval.classifier import load_cnn, classify_images

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    device = torch.device("cpu")

    g_path = os.path.join(run_dir, "G.pth")
    d_path = os.path.join(run_dir, "D.pth")
    if not os.path.exists(g_path) or not os.path.exists(d_path):
        print(f"Checkpoints not found in {run_dir} for accept/reject plot.")
        return

    config_path = os.path.join(run_dir, "config.json")
    disc_type = "bce"
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get("loss") == "prdiv":
            disc_type = "chi2"

    G = Generator(784)
    G.load_state_dict(torch.load(g_path, map_location=device, weights_only=True))
    G.eval()

    D = Discriminator(784)
    D.load_state_dict(torch.load(d_path, map_location=device, weights_only=True))
    D.eval()

    torch.manual_seed(42)
    z = torch.randn(n_eval, 100)
    with torch.no_grad():
        x = G(z)
        if disc_type == "bce":
            logit = D(x, logits=True).squeeze(-1)
            r = torch.exp(logit)
        else:
            T = D(x, logits=True).squeeze(-1)
            r = torch.clamp(T / 2 + 1, min=0)

    # Sort by ratio r
    sorted_idx = torch.argsort(r)
    lowest_idx = sorted_idx[:64]
    highest_idx = sorted_idx[-64:]

    # Save 8x8 grids
    x_imgs = (x.reshape(-1, 1, 28, 28) + 1) / 2
    torchvision.utils.save_image(x_imgs[highest_idx], "figures/obrs_highest_r.png", nrow=8)
    torchvision.utils.save_image(x_imgs[lowest_idx], "figures/obrs_lowest_r.png", nrow=8)

    # Scatter of r vs CNN confidence
    try:
        cnn = load_cnn(device)
        _, confs, _ = classify_images(x_imgs, cnn, device)
        r_np = r.numpy()
        confs_np = confs.numpy()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(confs_np, r_np, alpha=0.4, c="teal", edgecolors="none")
        ax.set_xlabel("CNN Classifier Max Confidence", fontsize=11)
        ax.set_ylabel("Estimated Density Ratio r(x)", fontsize=11)
        ax.set_title("Density Ratio vs. Digit Ambiguity", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved OBRS accept/reject figures to {out_path}, figures/obrs_highest_r.png, figures/obrs_lowest_r.png")
    except Exception as e:
        print(f"Scatter plot failed: {e}")


def make_tables(results, out_csv="tables/main.csv", out_tex="tables/main.tex"):
    """Aggregate metrics over seeds into CSV and LaTeX tables (mean +- std)."""
    import pandas as pd
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    rows = []
    for r in results:
        path = r.get("_file_path", "")
        run_name = os.path.basename(os.path.dirname(path))
        rows.append({
            "run": run_name,
            "FID": r.get("fid"),
            "Precision": r.get("vgg_precision"),
            "Recall": r.get("vgg_recall"),
            "Density": r.get("vgg_density"),
            "Coverage": r.get("vgg_coverage"),
            "CNN_P": r.get("cnn_precision"),
            "CNN_R": r.get("cnn_recall"),
            "LowConfShare": r.get("cnn", {}).get("low_conf_share"),
            "EvalTime": r.get("eval_time_seconds"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved main results table to {out_csv}")

    try:
        df.to_latex(out_tex, index=False, float_format="%.4f", na_rep="-")
        print(f"Saved LaTeX table to {out_tex}")
    except Exception:
        # Fallback manual LaTeX table generator if pandas to_latex fails
        with open(out_tex, "w") as f:
            f.write("\\begin{tabular}{" + "l" * len(df.columns) + "}\n\\hline\n")
            f.write(" & ".join(df.columns) + " \\\\\n\\hline\n")
            for _, r in df.iterrows():
                vals = [str(round(v, 4)) if isinstance(v, (int, float)) and not np.isnan(v) else str(v) for v in r]
                f.write(" & ".join(vals) + " \\\\\n")
            f.write("\\hline\n\\end{tabular}\n")
        print(f"Saved LaTeX table to {out_tex} (manual fallback)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    args = parser.parse_args()

    results = find_all_results(args.runs_dir)
    print(f"Found {len(results)} evaluated results.")

    if results:
        plot_pr_plane(results)
        plot_per_digit(results)
        make_tables(results)
    else:
        print("No evaluation JSONs found yet. Run eval/evaluate.py first.")

    # Try generating accept/reject if baseline exists
    for candidate in ["runs/e2_baseline_seed0", "runs/smoke_bce"]:
        if os.path.exists(os.path.join(candidate, "G.pth")):
            plot_obrs_accept_reject(candidate)
            break


if __name__ == "__main__":
    main()
