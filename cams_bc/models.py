"""Two pointwise learners with one interface: fit(X, y, w, X_val, y_val, w_val) and predict(X).
XGBoost (GPU hist, early stopping on the validation stations) and a multi-output MLP in torch."""
import copy

import numpy as np
import xgboost as xgb

XGB_DEFAULT = {"max_depth": 8, "learning_rate": 0.05, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
               "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0, "max_bin": 256}


class XGBModel:
    def __init__(self, params: dict = None, n_rounds: int = 3000, early_stopping: int = 100, device: str = "cuda:0", seed: int = 0):
        self.params = {**XGB_DEFAULT, **(params or {})}
        self.n_rounds, self.early, self.device, self.seed = n_rounds, early_stopping, device, seed
        self.booster, self.best_iteration = None, None

    def fit(self, X, y, w, X_val, y_val, w_val):
        p = {**self.params, "objective": "reg:squarederror", "tree_method": "hist", "device": self.device, "seed": self.seed,
             "verbosity": 0}
        dtr = xgb.DMatrix(X, label=y, weight=w, missing=np.nan)
        dva = xgb.DMatrix(X_val, label=y_val, weight=w_val, missing=np.nan)
        self.booster = xgb.train(p, dtr, self.n_rounds, evals=[(dva, "val")], early_stopping_rounds=self.early, verbose_eval=False)
        self.best_iteration = int(self.booster.best_iteration)
        return self

    def predict(self, X):
        return self.booster.predict(xgb.DMatrix(X, missing=np.nan), iteration_range=(0, self.best_iteration + 1))

    def importance(self, names: list) -> dict:
        g = self.booster.get_score(importance_type="gain")
        return {names[int(k[1:])]: v for k, v in g.items()}


MLP_DEFAULT = {"n_layers": 3, "width": 256, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 4096,
               "activation": "gelu", "lr_schedule": "plateau", "max_epochs": 60, "patience": 8}


class MLPModel:
    """Multi-output (one column per target) MLP with weighted MSE. Features are median-filled and
    standardised with statistics of the training rows only."""

    def __init__(self, params: dict = None, device: str = "cuda:0", seed: int = 0):
        self.p = {**MLP_DEFAULT, **(params or {})}
        self.device, self.seed = device, seed
        self.net, self.mu, self.sd, self.med, self.history = None, None, None, None, []

    def _prep(self, X, fit=False):
        X = np.asarray(X, np.float32)
        if fit:
            self.med = np.nanmedian(X, axis=0)
            self.med = np.where(np.isfinite(self.med), self.med, 0.0).astype(np.float32)
        X = np.where(np.isfinite(X), X, self.med)
        if fit:
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return (X - self.mu) / self.sd

    def _build(self, n_in, n_out):
        import torch.nn as nn
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[self.p["activation"]]
        layers, d = [], n_in
        for _ in range(int(self.p["n_layers"])):
            layers += [nn.Linear(d, int(self.p["width"])), act(), nn.Dropout(float(self.p["dropout"]))]
            d = int(self.p["width"])
        layers.append(nn.Linear(d, n_out))
        return nn.Sequential(*layers)

    def fit(self, X, y, w, X_val, y_val, w_val):
        import torch
        torch.manual_seed(self.seed)
        y, w, y_val, w_val = (np.asarray(a, np.float32) for a in (y, w, y_val, w_val))
        if y.ndim == 1:
            y, w, y_val, w_val = y[:, None], w[:, None], y_val[:, None], w_val[:, None]
        Xt = torch.tensor(self._prep(X, fit=True), device=self.device)
        Xv = torch.tensor(self._prep(X_val), device=self.device)
        yt, wt = torch.tensor(y, device=self.device), torch.tensor(w, device=self.device)
        yv, wv = torch.tensor(y_val, device=self.device), torch.tensor(w_val, device=self.device)
        self.net = self._build(Xt.shape[1], yt.shape[1]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(self.p["lr"]), weight_decay=float(self.p["weight_decay"]))
        if self.p["lr_schedule"] == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(self.p["max_epochs"]))
        else:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        bs, n = int(self.p["batch_size"]), Xt.shape[0]
        best, best_state, bad = np.inf, None, 0
        for epoch in range(int(self.p["max_epochs"])):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                out = self.net(Xt[idx])
                loss = (wt[idx] * (out - yt[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(sum(((wv[i:i + 65536] * (self.net(Xv[i:i + 65536]) - yv[i:i + 65536]) ** 2).sum()) for i in range(0, Xv.shape[0], 65536)) / (Xv.shape[0] * yv.shape[1]))
            self.history.append(vl)
            if self.p["lr_schedule"] == "cosine":
                sched.step()
            else:
                sched.step(vl)
            if vl < best - 1e-6:
                best, bad, best_state = vl, 0, copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= int(self.p["patience"]):
                    break
        self.net.load_state_dict(best_state)
        self.best_val, self.epochs = best, len(self.history)
        return self

    def predict(self, X):
        import torch
        Xt = torch.tensor(self._prep(X), device=self.device)
        self.net.eval()
        with torch.no_grad():
            out = torch.cat([self.net(Xt[i:i + 65536]) for i in range(0, Xt.shape[0], 65536)]).cpu().numpy()
        return out[:, 0] if out.shape[1] == 1 else out
