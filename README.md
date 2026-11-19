# Where does a small GAN lose precision, and where does it lose recall?

A capacity-constrained MLP generator, frozen architecture, MNIST. The question is not how to
build a better GAN but **which intervention fixes which failure** — fidelity or mode coverage —
and whether the mechanism proposed for one actually explains the change observed.

235 configurations evaluated on GPU. Coursework for the Data Science Lab, M2 IASD
(Université Paris-Dauphine – PSL), October–November 2026.

## Results

Measured with Inception-v3 FID and VGG-16 precision/recall at *k* = 5.

| Setup | Inception FID ↓ | Precision ↑ | Recall ↑ |
|---|---|---|---|
| E1 — vanilla GAN, β₁ = 0.9 | 43.64 | 0.379 | 0.449 |
| E2 — stable baseline, β₁ = 0.5 | 51.34 | 0.562 | 0.335 |
| E8 — R₁ regularization + TTUR | 23.58 | 0.555 | **0.553** |
| E8 + OBRS rejection sampling (*K* = 10) | **23.04** | **0.566** | 0.547 |

Three findings worth stating plainly:

- **β₁ = 0.9 is not a neutral choice.** One seed in three collapsed into pure noise (FID 389).
  Lowering β₁ to 0.5 gave stability across every seed — but bought fidelity by dropping the
  harder digits, 2 and 8. Stability and coverage traded against each other.
- **R₁ + TTUR recovered coverage without touching the architecture**: recall 0.335 → 0.553,
  a 65% relative gain, with FID cut by more than half.
- **Rejection sampling acts at inference, on a different limitation.** Applied to the weak
  baseline it made things worse (FID 53.99); applied to E8 it pruned residual boundary
  artifacts and gave the lowest FID of the study. The two interventions are not interchangeable.

## A note on FID scales

FID is only comparable within one feature space. The numbers above use Inception-v3 (2048-d,
ImageNet features), where an MLP on 28×28 grayscale MNIST typically scores 40–55. Another common
choice is a 128-d MNIST-CNN feature space, where the same models score 5–17 — ours reached 6.22
there. Both are reported because neither alone is meaningful.

## Layout

| Path | Content |
|---|---|
| `model.py`, `losses.py`, `train.py` | Generator and discriminator, the loss variants, training |
| `eval/` | FID, precision/recall at *k*, the feature extractors, a real-vs-real control |
| `analysis/` | Result collection, checkpoint inspection, figures |
| `experiments/` | The configuration of each run, E1 through E15 |
| `results/` | `all_results.csv` — every run, every metric — and a summary |
| `figures/` | Precision–recall plane, per-digit coverage, accepted and rejected samples |
| `checkpoints/G.pth` | The submitted generator (SN-MLP + EMA, 500 epochs, seed 0) |
| `report/` | The written report and the slides |
| `scripts/` | Slurm job scripts used on the cluster |

The datasets, the evaluation caches and the sample dumps are not redistributed.

## Reproducing

```bash
pip install -r requirements.txt -r requirements-eval.txt
python train.py --config experiments/e8_r1_ttur.txt
python eval/evaluate.py --ckpt checkpoints/G.pth
```

Training is deterministic under a fixed seed: the EMA snapshots at 150 and 300 epochs reproduce
the earlier runs exactly, which is what let the training-length curve be read as a single curve
rather than six separate experiments.

## Authors

Carried out for the Data Science Lab course of the M2 IASD, Université Paris-Dauphine – PSL,
by Lynda Mellissa Yazi and Raphael Breaud.

## License

MIT, see [LICENSE](LICENSE). The course materials are not covered by it.
