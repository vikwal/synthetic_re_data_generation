"""Model architectures, training and persistence for the pre-trained model tab.

Architectures (all map a window (lookback, features) -> capacity fraction):
  linear : ridge regression on the flattened window (numpy, no extra dependencies)
  gru    : stacked GRU                                   (needs PyTorch)
  tcn    : dilated causal temporal convolutional network (needs PyTorch)
  tft    : compact Temporal-Fusion-style network: variable selection, gated residual
           networks, LSTM encoder and causal self-attention (needs PyTorch). This is a
           from-scratch, lightweight variant, not the pytorch-forecasting implementation.

PyTorch is imported lazily so the rest of the app runs without it.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

ProgressFn = Optional[Callable[[float, str], None]]

ARCH_INFO = {
    "linear": ("Linear regression",
               "Fast and easy to interpret. Weighs each of the last hours of weather by a fixed factor to estimate output. "
               "Trains in seconds; a good baseline, but it can't capture complex behaviour such as the flat top of a power curve.", False),
    "gru": ("GRU",
            "A neural network that reads the weather hour by hour and remembers what matters. "
            "Copes well with changing conditions; moderate training time.", True),
    "tcn": ("TCN",
            "A neural network that scans the weather history at several time scales at once. "
            "Picks up longer patterns and usually trains faster than a GRU.", True),
    "tft": ("TFT",
            "Temporal Fusion Transformer, in a smaller form. Learns which weather inputs matter most, "
            "then combines memory of recent hours with attention to the key moments. "
            "The most flexible option and usually the most accurate, but the slowest to train.", True),
}

# What early stopping watches (all in %, lower is better).
MONITORS = {"val_rmse": "Validation RMSE", "val_mae": "Validation MAE", "train_loss": "Training RMSE"}
LIMITS = {"lookback": (1, 168), "horizon": (1, 72), "epochs": (1, 300), "hidden": (8, 256), "layers": (1, 4),
          "batch_size": (16, 2048), "patience": (1, 50)}


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def availability() -> List[dict]:
    has_torch = torch_available()
    out = []
    for key, (label, desc, needs_torch) in ARCH_INFO.items():
        ok = has_torch or not needs_torch
        out.append({"id": key, "label": label, "description": desc, "available": ok,
                    "reason": None if ok else "Requires PyTorch: pip install torch"})
    return out


def clamp_hp(hp: dict) -> dict:
    """Validated hyper-parameters with sane defaults."""
    out = {
        "lookback": 24, "epochs": 30, "hidden": 64, "layers": 2, "dropout": 0.1, "lr": 1e-3,
        "batch_size": 128, "val_frac": 0.2, "ridge_alpha": 1.0, "seed": 42, "patience": 6, "weight_decay": 1e-5,
        "horizon": 1, "scaling": "standard",
        "early_stopping": True, "monitor": "val_rmse", "min_delta": 0.001,
    }
    out.update({k: v for k, v in (hp or {}).items() if v is not None})
    for k, (lo, hi) in LIMITS.items():
        out[k] = int(min(max(int(out[k]), lo), hi))
    if out.get("scaling") not in ("standard", "minmax", "none"):
        out["scaling"] = "standard"
    if out.get("monitor") not in MONITORS:
        out["monitor"] = "val_rmse"
    out["early_stopping"] = bool(out.get("early_stopping", True))
    out["min_delta"] = float(max(float(out["min_delta"] if out.get("min_delta") is not None else 0.001), 0.0))
    out["hidden"] = max(8, (out["hidden"] // 4) * 4)          # attention heads need hidden % 4 == 0
    out["dropout"] = float(min(max(float(out["dropout"]), 0.0), 0.6))
    out["lr"] = float(min(max(float(out["lr"]), 1e-6), 1.0))
    out["val_frac"] = float(min(max(float(out["val_frac"]), 0.05), 0.5))
    out["ridge_alpha"] = float(max(float(out["ridge_alpha"]), 1e-8))
    out["weight_decay"] = float(max(float(out["weight_decay"]), 0.0))
    return out


# ------------------------------------------------------------------ linear (numpy)
class LinearModel:
    arch = "linear"
    ext = ".npz"

    def __init__(self, w: np.ndarray, b, lookback: int, n_features: int):
        w = np.asarray(w, dtype=np.float64)
        self.w = w if w.ndim == 2 else w[:, None]                         # (L*F, horizon); older files hold (L*F,)
        self.b = np.atleast_1d(np.asarray(b, dtype=np.float64))
        self.lookback, self.n_features = int(lookback), int(n_features)

    @property
    def horizon(self) -> int:
        return self.w.shape[1]

    @staticmethod
    def fit(Xtr: np.ndarray, ytr: np.ndarray, alpha: float, prior: Optional["LinearModel"] = None) -> "LinearModel":
        """Ridge solution (one output per forecast hour). With `prior`, shrinks towards the pre-trained weights
        instead of zero, so fine-tuning on little real data stays close to what the synthetic data taught."""
        n, L, F = Xtr.shape
        A = Xtr.reshape(n, -1).astype(np.float64)
        y = ytr.astype(np.float64).reshape(n, -1)                          # (N, horizon)
        if prior is None:
            xm, ym = A.mean(axis=0), y.mean(axis=0)
            Ac, yc = A - xm, y - ym
            w = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(A.shape[1]), Ac.T @ yc)
            b = ym - xm @ w
        else:
            w0, b0 = prior.w, prior.b
            resid = y - (A @ w0 + b0)
            xm, rm = A.mean(axis=0), resid.mean(axis=0)
            Ac, rc = A - xm, resid - rm
            dw = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(A.shape[1]), Ac.T @ rc)
            w = w0 + dw
            b = b0 + rm - xm @ dw
        return LinearModel(w, b, L, F)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = (X.reshape(len(X), -1).astype(np.float64) @ self.w + self.b).astype(np.float32)
        return out[:, 0] if self.horizon == 1 else out

    def save(self, path: Path) -> None:
        np.savez(path, w=self.w, b=self.b, lookback=self.lookback, n_features=self.n_features)

    @staticmethod
    def load(path: Path) -> "LinearModel":
        z = np.load(path)
        return LinearModel(z["w"], z["b"], int(z["lookback"]), int(z["n_features"]))


# ------------------------------------------------------------------ torch models
def _build_torch(arch: str, n_features: int, lookback: int, hp: dict, n_out: int = 1):
    import torch
    from torch import nn
    import torch.nn.functional as F

    hidden, layers, dropout = int(hp["hidden"]), int(hp["layers"]), float(hp["dropout"])
    squeeze = (lambda t: t.squeeze(-1)) if n_out == 1 else (lambda t: t)       # (B,) for one output, else (B, horizon)

    if arch == "gru":
        class GRUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                                  dropout=dropout if layers > 1 else 0.0)
                self.head = nn.Linear(hidden, n_out)

            def forward(self, x):
                out, _ = self.rnn(x)
                return squeeze(self.head(out[:, -1]))
        return GRUNet()

    if arch == "tcn":
        class Block(nn.Module):
            def __init__(self, cin, cout, dilation, kernel=3):
                super().__init__()
                self.pad = (kernel - 1) * dilation
                self.c1 = nn.Conv1d(cin, cout, kernel, dilation=dilation)
                self.c2 = nn.Conv1d(cout, cout, kernel, dilation=dilation)
                self.drop = nn.Dropout(dropout)
                self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

            def forward(self, x):
                y = self.drop(F.relu(self.c1(F.pad(x, (self.pad, 0)))))
                y = self.drop(F.relu(self.c2(F.pad(y, (self.pad, 0)))))
                return F.relu(y + self.down(x))

        class TCN(nn.Module):
            def __init__(self):
                super().__init__()
                chans = [n_features] + [hidden] * layers
                self.blocks = nn.Sequential(*[Block(chans[i], chans[i + 1], 2 ** i) for i in range(layers)])
                self.head = nn.Linear(hidden, n_out)

            def forward(self, x):
                y = self.blocks(x.transpose(1, 2))             # (B, H, L)
                return squeeze(self.head(y[:, :, -1]))
        return TCN()

    if arch == "tft":
        class GRN(nn.Module):
            """Gated residual network."""
            def __init__(self, d):
                super().__init__()
                self.fc1, self.fc2 = nn.Linear(d, d), nn.Linear(d, d)
                self.gate = nn.Linear(d, 2 * d)
                self.norm = nn.LayerNorm(d)
                self.drop = nn.Dropout(dropout)

            def forward(self, x):
                h = self.drop(self.fc2(F.elu(self.fc1(x))))
                return self.norm(x + F.glu(self.gate(h), dim=-1))

        class TFT(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.ModuleList([nn.Linear(1, hidden) for _ in range(n_features)])
                self.select = nn.Linear(n_features, n_features)          # variable-selection weights
                self.grn_in = GRN(hidden)
                self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
                self.grn_lstm = GRN(hidden)
                self.attn = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
                self.norm = nn.LayerNorm(hidden)
                self.grn_out = GRN(hidden)
                self.head = nn.Linear(hidden, n_out)

            def forward(self, x):
                B, L, Fdim = x.shape
                w = torch.softmax(self.select(x), dim=-1)                              # (B, L, F)
                emb = torch.stack([m(x[..., i:i + 1]) for i, m in enumerate(self.embed)], dim=2)  # (B, L, F, H)
                z = self.grn_in((emb * w.unsqueeze(-1)).sum(dim=2))                     # (B, L, H)
                h, _ = self.lstm(z)
                h = self.grn_lstm(h + z)
                mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
                a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
                h = self.grn_out(self.norm(h + a))
                return squeeze(self.head(h[:, -1]))
        return TFT()

    raise ValueError(f"Unknown architecture '{arch}'")


class TorchModel:
    ext = ".pt"

    def __init__(self, arch: str, net, hp: dict, n_features: int, lookback: int, n_out: int = 1):
        self.arch, self.net, self.hp = arch, net, hp
        self.n_features, self.lookback, self.n_out = int(n_features), int(lookback), int(n_out)

    def predict(self, X: np.ndarray, batch: int = 1024) -> np.ndarray:
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                out.append(self.net(torch.from_numpy(X[i:i + batch])).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)

    def save(self, path: Path) -> None:
        import torch
        torch.save({"arch": self.arch, "state": self.net.state_dict(), "hp": self.hp,
                    "n_features": self.n_features, "lookback": self.lookback, "n_out": self.n_out}, path)

    @staticmethod
    def load(path: Path) -> "TorchModel":
        import torch
        blob = torch.load(path, map_location="cpu")
        n_out = int(blob.get("n_out", 1))
        net = _build_torch(blob["arch"], blob["n_features"], blob["lookback"], blob["hp"], n_out)
        net.load_state_dict(blob["state"])
        return TorchModel(blob["arch"], net, blob["hp"], blob["n_features"], blob["lookback"], n_out)


def load_model(arch: str, path: Path):
    return LinearModel.load(path) if arch == "linear" else TorchModel.load(path)


def weights_ext(arch: str) -> str:
    return ".npz" if arch == "linear" else ".pt"


# ------------------------------------------------------------------ training
class EarlyStopper:
    """Tracks the monitored metric (the number shown while training, lower is better). An epoch only counts as an improvement if it
    beats the best so far by more than `min_delta`; after `patience` epochs without one, `update` says stop."""

    def __init__(self, monitor: str = "val_rmse", min_delta: float = 0.0, patience: int = 6, enabled: bool = True):
        self.monitor, self.min_delta, self.patience, self.enabled = monitor, float(min_delta), int(patience), bool(enabled)
        self.best, self.stale = float("inf"), 0

    def update(self, val_rmse: float, val_mae: float, train_rmse: float) -> Tuple[bool, bool]:
        """Returns (improved, stop)."""
        score = {"val_rmse": val_rmse, "val_mae": val_mae, "train_loss": train_rmse}[self.monitor]
        if score < self.best - self.min_delta:
            self.best, self.stale = score, 0
            return True, False
        self.stale += 1
        return False, self.enabled and self.stale >= self.patience



def _report(progress: ProgressFn, frac: float, msg: str) -> None:
    if progress:
        progress(frac, msg)


def train(arch: str, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, hp: dict,
          progress: ProgressFn = None, init=None, freeze: bool = False,
          restore_best: bool = True) -> Tuple[object, Dict[str, List[float]]]:
    """Fit a model. `init` is a pre-trained model of the same architecture (fine-tuning);
    `freeze` trains only the output layer (torch models). `restore_best=False` keeps the last epoch's weights instead of
    the best one (used for the refit on all data). `history["best_epoch"]` is the 1-based epoch with the best monitored score."""
    if arch not in ARCH_INFO:
        raise ValueError(f"Unknown architecture '{arch}'")
    L, F = Xtr.shape[1], Xtr.shape[2]
    n_out = 1 if ytr.ndim == 1 else ytr.shape[1]

    if arch == "linear":
        _report(progress, 0.2, "Fitting ridge regression")
        model = LinearModel.fit(Xtr, ytr, hp["ridge_alpha"], prior=init)
        mse_tr = float(np.mean((model.predict(Xtr) - ytr) ** 2))
        mse_va = float(np.mean((model.predict(Xva) - yva) ** 2))
        _report(progress, 0.9, "Evaluating")
        return model, {"train_loss": [mse_tr], "val_loss": [mse_va]}

    if not torch_available():
        raise RuntimeError("PyTorch is not installed. Run `pip install torch` and restart the server "
                           "to train GRU, TCN or TFT models.")
    import torch
    from torch import nn

    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])
    if init is not None:
        net = init.net
    else:
        net = _build_torch(arch, F, L, hp, n_out)
    if freeze:
        for name, p in net.named_parameters():
            p.requires_grad = name.startswith("head")
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
    n, bs = len(xt), hp["batch_size"]
    hist = {"train_loss": [], "val_loss": [], "val_mae": [], "best_epoch": 1}
    best_state = copy.deepcopy(net.state_dict())
    stopper = EarlyStopper(hp["monitor"], hp["min_delta"], hp["patience"], hp["early_stopping"])

    for epoch in range(hp["epochs"]):
        net.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += float(loss.item()) * len(idx)
        net.eval()
        se = ae = 0.0
        cnt = 0
        with torch.no_grad():
            for i in range(0, len(xv), 1024):
                diff = net(xv[i:i + 1024]) - yv[i:i + 1024]
                se += float((diff ** 2).sum())
                ae += float(diff.abs().sum())
                cnt += diff.numel()
        val, val_mae, train = se / cnt, ae / cnt, total / n
        hist["train_loss"].append(train)
        hist["val_loss"].append(val)
        hist["val_mae"].append(val_mae)
        _report(progress, (epoch + 1) / hp["epochs"],
                f"Epoch {epoch + 1}/{hp['epochs']} · val RMSE {100 * val ** 0.5:.2f}%")
        improved, stop = stopper.update(100 * val ** 0.5, 100 * val_mae, 100 * train ** 0.5)
        if improved:
            best_state = copy.deepcopy(net.state_dict())
            hist["best_epoch"] = epoch + 1
        if stop:
            _report(progress, 1.0, f"Early stop at epoch {epoch + 1} ({MONITORS[hp['monitor']].lower()} stopped improving)")
            break
    if restore_best:
        net.load_state_dict(best_state)
    for p in net.parameters():
        p.requires_grad = True
    return TorchModel(arch, net, hp, F, L, n_out), hist
