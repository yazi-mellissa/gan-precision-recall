"""Train MNIST classifier and extract features/predictions."""
import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
CNN_PATH = os.path.join(ASSETS_DIR, "mnist_cnn.pt")


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        feat = x  # 128-d penultimate features
        x = self.dropout2(x)
        x = self.fc2(x)
        return x, feat

    def predict(self, x):
        """Return logits and 128-d features."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def train_cnn(epochs=6, lr=1e-3, batch_size=64, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.environ.get("DATA", "data")
    download = data_path == "data"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(data_path, train=True, download=download, transform=transform)
    test_ds = datasets.MNIST(data_path, train=False, download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.shape[0]
        acc = correct / total
        print(f"Epoch {epoch}/{epochs}: test acc = {acc:.4f}")
        if acc >= 0.99:
            print(f"Target accuracy reached (acc = {acc:.4f} >= 0.99) at epoch {epoch}!")
            break

    if acc < 0.99:
        raise RuntimeError(f"CNN test accuracy {acc:.4f} < 0.99, aborting!")

    os.makedirs(ASSETS_DIR, exist_ok=True)
    torch.save(model.state_dict(), CNN_PATH)
    print(f"Saved CNN to {CNN_PATH} (acc={acc:.4f})")
    return model


def load_cnn(device=None):
    if device is None:
        device = torch.device("cpu")
    model = MnistCNN().to(device)
    if not os.path.exists(CNN_PATH):
        raise FileNotFoundError(f"Trained CNN checkpoint not found at {CNN_PATH}. Run eval/classifier.py first.")
    model.load_state_dict(torch.load(CNN_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def classify_images(images, model, device, batch_size=256):
    """Classify 28x28 images (tensor [N,1,28,28] or [N,784] in [0,1]).
    Returns: predictions (N,), confidences (N,), features (N,128)."""
    model.eval()
    if images.dim() == 2:
        images = images.view(-1, 1, 28, 28)
    # Normalize to standard MNIST stats
    images = (images - 0.1307) / 0.3081

    all_preds, all_confs, all_feats = [], [], []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            logits, feats = model(batch)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.append(preds.cpu())
            all_confs.append(confs.cpu())
            all_feats.append(feats.cpu())

    return torch.cat(all_preds), torch.cat(all_confs), torch.cat(all_feats)


def digit_stats(preds, confs, conf_threshold=0.9):
    """Compute digit histogram over confident samples and low-confidence share."""
    confident = confs >= conf_threshold
    hist = [0] * 10
    for d in range(10):
        hist[d] = int(((preds == d) & confident).sum().item())
    total_conf = int(confident.sum().item())
    total_samples = len(preds)
    low_conf_share = 1.0 - (total_conf / total_samples) if total_samples > 0 else 0.0

    if total_conf > 0:
        probs = [c / total_conf for c in hist if c > 0]
        entropy = -sum(p * math.log(p) for p in probs)
    else:
        entropy = 0.0

    return {
        "digit_hist": hist,
        "n_confident": total_conf,
        "low_conf_share": float(low_conf_share),
        "hist_entropy": float(entropy),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cnn(epochs=args.epochs, lr=args.lr, device=device)
