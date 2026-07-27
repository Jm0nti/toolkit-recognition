"""CNN pequeña en PyTorch para clasificación de recortes.

Arquitectura (~200k params, entrena rápido en CPU/GPU):

    Conv(32) → BN → ReLU → Pool
    Conv(64) → BN → ReLU → Pool
    Conv(128) → BN → ReLU → Pool
    GlobalAvgPool → FC(128) → Dropout(0.3) → FC(num_classes)

Entrada: RGB 96×96. Se aplica una aumentación ligera (flips, rotaciones,
color jitter) durante el entrenamiento.

Interfaz común a los tres clasificadores:
    train(X_bgr, y, class_names, X_val=None, y_val=None) -> model
    predict(model, X_bgr)                                 -> np.ndarray
    save(model, path)                                     -> None
    load(path)                                            -> model
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


NAME = "cnn"
INPUT_SIZE = 96


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
class ToolCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _preproc(img_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """BGR uint8 arbitrario → RGB float32 [0,1] tamaño fijo con letterbox."""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.float32)
    escala = size / max(h, w)
    nh, nw = max(1, int(h * escala)), max(1, int(w * escala))
    redim = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    lienzo = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = redim
    rgb = cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)


def _augment(rgb: np.ndarray) -> np.ndarray:
    """Aumentación ligera en NumPy (flips + jitter). Complementa el pipeline previo."""
    if np.random.rand() < 0.5:
        rgb = rgb[:, ::-1, :].copy()
    if np.random.rand() < 0.3:
        rgb = np.rot90(rgb, k=np.random.randint(1, 4)).copy()
    if np.random.rand() < 0.5:
        rgb = np.clip(rgb + (np.random.rand() - 0.5) * 0.2, 0, 1).astype(np.float32)
    return rgb


class CropsDataset(Dataset):
    def __init__(self, X_bgr, y, augment: bool = False):
        self.X = X_bgr
        self.y = y
        self.aug = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        rgb = _preproc(self.X[idx])
        if self.aug:
            rgb = _augment(rgb)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        return tensor, int(self.y[idx])


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #
def _pick_device() -> str:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            return "cuda"
        except Exception:
            pass
    return "cpu"


def train(X_bgr: list[np.ndarray],
          y: np.ndarray,
          class_names: list[str],
          X_val: list[np.ndarray] | None = None,
          y_val: np.ndarray | None = None,
          epochs: int = 25,
          batch_size: int = 32,
          lr: float = 1e-3,
          weight_decay: float = 1e-4) -> dict:
    """
    Entrena la CNN. Si se pasa X_val/y_val, hace early-stopping por accuracy
    de validación (paciencia=6 epochs sin mejorar).
    """
    device = _pick_device()
    print(f"[{NAME}] device={device}  train={len(X_bgr)}  clases={len(class_names)}")

    # class-weighted CE para compensar desbalance
    counts = np.bincount(y, minlength=len(class_names)).astype(np.float32)
    pesos = counts.sum() / (len(class_names) * (counts + 1e-6))
    pesos_tensor = torch.tensor(pesos, dtype=torch.float32, device=device)

    model = ToolCNN(len(class_names)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=pesos_tensor)

    tr_loader = DataLoader(CropsDataset(X_bgr, y, augment=True),
                           batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        va_loader = DataLoader(CropsDataset(X_val, y_val, augment=False),
                               batch_size=batch_size, shuffle=False, num_workers=0)

    best_val, best_state, paciencia_actual = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_acum, n = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()
            loss_acum += loss.item() * xb.size(0)
            n += xb.size(0)
        tr_loss = loss_acum / max(1, n)

        if va_loader is not None:
            va_acc = _accuracy(model, va_loader, device)
            print(f"  epoch {epoch:>2}/{epochs}  train_loss={tr_loss:.4f}  val_acc={va_acc:.3f}")
            if va_acc > best_val:
                best_val, best_state, paciencia_actual = va_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                paciencia_actual += 1
                if paciencia_actual >= 6:
                    print(f"  early stop en epoch {epoch}")
                    break
        else:
            print(f"  epoch {epoch:>2}/{epochs}  train_loss={tr_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    t_train = time.time() - t0

    return {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "class_names": class_names,
        "duracion_train_s": t_train,
        "best_val_acc": None if best_val < 0 else best_val,
    }


def _accuracy(model, loader, device) -> float:
    model.eval()
    ok, tot = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            ok += (pred == yb).sum().item()
            tot += yb.numel()
    return ok / max(1, tot)


def predict(model: dict, X_bgr: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
    device = _pick_device()
    class_names = model["class_names"]
    net = ToolCNN(len(class_names)).to(device)
    net.load_state_dict(model["state_dict"])
    net.eval()

    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(X_bgr), batch_size):
            batch = [_preproc(img) for img in X_bgr[i:i + batch_size]]
            xb = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).contiguous().to(device)
            out = net(xb).argmax(dim=1).cpu().numpy()
            preds.extend(out.tolist())
    return np.asarray(preds, dtype=np.int64)


def save(model: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)
    print(f"[{NAME}] guardado en {path}")


def load(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
