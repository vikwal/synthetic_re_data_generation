"""Optuna search spaces. The feature set and (by default) the target form come from the predictor
study and are NOT searched - what was measured before costs no trials (solar_bc/hpo.py principle).

v3 (2026-10-13): ranges widened where the v2 top-10 trials sat at an edge (rule: hit a bound ->
widen, stop, restart with the best v2 trials enqueued). XGB: max_depth 12 -> 16, learning_rate
floor 0.01 -> 0.005, min_child_weight floor 1 -> 0.1, subsample floor 0.5 -> 0.2, colsample floor
0.3 -> 0.2, reg_lambda/alpha ceilings x3, max_bin + 1024. MLP: n_layers 5 -> 8, dropout 0.3 -> 0.5,
lr floor 1e-4 -> 2e-5, weight_decay floor 1e-6 -> 1e-8, batch_size floor 1024 -> 256. gamma keeps
its natural floor 0 (= off).

v4 (2026-10-13, after the independent review): v3 top-10 trials touched max_depth 16, min_child_weight
~200 (97 % of the log range), colsample 0.2 and n_layers 8 -> max_depth 24, min_child_weight 1000,
colsample floor 0.1, n_layers 12 (v6: 20 after the v5 best trial sat at 12). Pruning changed as well: the MedianPruner compares a trial's BEST
intermediate value with the median at the current step, and fold 0 is systematically ~0.03 easier
than folds 1/2, so with per-fold reports nothing was ever pruned in v3. v4 reports the running mean
over the folds in FOLD_ORDER (hardest first), which makes the intermediate values comparable."""
import optuna

from .targets import FORMS

# Fold order for the HPO reports: hardest validation fold first (v3 mean objective per fold:
# fold 1 0.782, fold 2 0.779, fold 0 0.748), so the running mean falls over the steps and the
# pruner's "best intermediate value" is the latest, most complete one.
FOLD_ORDER = (1, 2, 0)


def study_name(model: str, version: str = "v1") -> str:
    return f"cams_bc_{model}_{version}"


def sample_xgb(trial: optuna.Trial, form: str = None) -> dict:
    p = {"max_depth": trial.suggest_int("max_depth", 4, 24),
         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
         "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 1000.0, log=True),
         "subsample": trial.suggest_float("subsample", 0.2, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 300.0, log=True),
         "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30.0, log=True),
         "gamma": trial.suggest_float("gamma", 0.0, 5.0),
         "max_bin": trial.suggest_categorical("max_bin", [256, 512, 1024])}
    p["form"] = form or trial.suggest_categorical("form", list(FORMS))
    return p


def sample_mlp(trial: optuna.Trial, form: str = None) -> dict:
    p = {"n_layers": trial.suggest_int("n_layers", 2, 20),
         "width": trial.suggest_int("width", 64, 1024, log=True),
         "dropout": trial.suggest_float("dropout", 0.0, 0.5),
         "lr": trial.suggest_float("lr", 2e-5, 1e-2, log=True),
         "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
         "batch_size": trial.suggest_int("batch_size", 256, 16384, log=True),
         "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
         "lr_schedule": trial.suggest_categorical("lr_schedule", ["plateau", "cosine"])}
    p["form"] = form or trial.suggest_categorical("form", list(FORMS))
    return p
