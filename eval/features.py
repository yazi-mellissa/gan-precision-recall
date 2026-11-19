"""Feature extraction: VGG16 fc2, Inception pool3, CNN penultimate."""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


_VGG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_vgg(layer="fc2_relu", device=None):
    """Load VGG16 feature extractor.
    fc2_relu: classifier[:5] = fc1, relu, dropout, fc2, relu (4096-d)
    fc2: classifier[:4] = fc1, relu, dropout, fc2 (4096-d)
    """
    if device is None:
        device = get_device()
    # No fallback to random weights: random VGG features would silently make every P/R number meaningless.
    # On an offline node, download the weights once on the login node (they are cached in ~/.cache/torch).
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    vgg.eval()
    if layer == "fc2_relu":
        extractor = nn.Sequential(
            vgg.features,
            vgg.avgpool,
            nn.Flatten(),
            vgg.classifier[:5]
        )
    elif layer == "fc2":
        extractor = nn.Sequential(
            vgg.features,
            vgg.avgpool,
            nn.Flatten(),
            vgg.classifier[:4]
        )
    else:
        raise ValueError(f"Unknown VGG layer: {layer}")
    extractor.eval()
    return extractor


def vgg_features(images, device=None, layer="fc2_relu", batch_size=64):
    """Extract VGG16 features from images (tensor N x 1 x 28 x 28 in [0, 1]).
    Returns numpy array of shape (N, 4096)."""
    if device is None:
        device = get_device()
    extractor = _load_vgg(layer, device)
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            if batch.dim() == 2:
                batch = batch.view(-1, 1, 28, 28)
            # Repeat grayscale to 3 channels
            batch = batch.repeat(1, 3, 1, 1)
            batch = _VGG_TRANSFORM(batch).to(device)
            feats = extractor(batch)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def load_pngs(folder, n=None, indices=None):
    """Load PNG images from folder into tensor of shape (N, 1, 28, 28) in [0, 1]."""
    if indices is not None:
        files = [os.path.join(folder, f"{i}.png") for i in indices]
    else:
        files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        if n is not None:
            files = files[:n]

    images = []
    for f in files:
        img = Image.open(f).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)
    return torch.tensor(np.stack(images)).unsqueeze(1)


def cache_features(images, cache_path, extractor_fn, **kwargs):
    """Extract features or load from .npy cache."""
    if os.path.exists(cache_path):
        return np.load(cache_path)
    feats = extractor_fn(images, **kwargs)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, feats)
    return feats


def export_real_pngs(out_dir="eval/assets/real_test_png", n=10000):
    """Export MNIST test set images as PNGs using torchvision save_image."""
    import torchvision
    data_path = os.environ.get("DATA", "data")
    download = data_path == "data"
    ds = torchvision.datasets.MNIST(
        root=data_path,
        train=False,
        download=download,
        transform=transforms.ToTensor()
    )
    os.makedirs(out_dir, exist_ok=True)
    n_export = min(n, len(ds))
    for i in range(n_export):
        img, _ = ds[i]  # shape (1, 28, 28) in [0, 1]
        torchvision.utils.save_image(img, os.path.join(out_dir, f"{i}.png"))
    print(f"Exported {n_export} real images to {out_dir}")


def get_pr_indices(n=9984, seed=1234):
    """Pick fixed subset of test image indices for P/R computation."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(10000)[:n]
    return sorted(indices.tolist())


if __name__ == "__main__":
    export_real_pngs()
    print("Done exporting real test PNGs.")
