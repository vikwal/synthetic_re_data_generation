"""Optuna search spaces. The feature set and (by default) the target form come from the predictor
study and are NOT searched - what was measured before costs no trials (solar_bc/hpo.py principle)."""
import optuna

from .targets import FORMS


def study_name(model: str, version: str = "v1") -> str:
    return f"cams_bc_{model}_{version}"


def sample_xgb(trial: optuna.Trial, form: str = None) -> dict:
    p = {"max_depth": trial.suggest_int("max_depth", 4, 12),
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
         "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 200.0, log=True),
         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
         "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
         "gamma": trial.suggest_float("gamma", 0.0, 5.0),
         "max_bin": trial.suggest_categorical("max_bin", [256, 512])}
    p["form"] = form or trial.suggest_categorical("form", list(FORMS))
    return p


def sample_mlp(trial: optuna.Trial, form: str = None) -> dict:
    p = {"n_layers": trial.suggest_int("n_layers", 2, 5),
         "width": trial.suggest_int("width", 64, 1024, log=True),
         "dropout": trial.suggest_float("dropout", 0.0, 0.3),
         "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
         "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
         "batch_size": trial.suggest_int("batch_size", 1024, 16384, log=True),
         "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
         "lr_schedule": trial.suggest_categorical("lr_schedule", ["plateau", "cosine"])}
    p["form"] = form or trial.suggest_categorical("form", list(FORMS))
    return p
