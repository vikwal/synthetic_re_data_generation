"""HPO search space + study naming for the TR models.

v2 studies (era5_bc_{model}_v2): continuous full space, TPE-sampled —
replaces the v1 categorical grid (kept in the DB for reference; the paper
default config is trial 0 of each v1 study).

Transformer hidden size is sampled as nheads * head_dim so the
divisibility constraint holds by construction (same pattern as
nwp_out_per_head in the forecasting_framework DCRNN HPO); bounds are chosen
so the product stays within the 16..128 range.
huber_delta and batch_size are deliberately NOT tuned: delta defines the
objective scale (val losses across trials would be incomparable), batch size
is absorbed by lr.
"""

import optuna

STUDY_VERSION = "v2"


def study_name(model: str) -> str:
    return f"era5_bc_{model}_{STUDY_VERSION}"


def sample_params(trial: optuna.Trial, space: dict, model: str) -> dict:
    """Draw one config from the full space. Model-specific keys:
    LSTM: hidden directly + num_layers; Transformer: nheads*head_dim + nblock."""
    p = {
        "lr": trial.suggest_float("lr", *_lh(space["lr"]), log=True),
        "dropout": trial.suggest_float("dropout", *_lh(space["dropout"])),
        "past_len": trial.suggest_int("past_len", *_lh(space["past_len"]),
                                      step=space["past_len"]["step"]),
        "static_hidden": trial.suggest_int("static_hidden",
                                           *_lh(space["static_hidden"])),
        "static_dropout": trial.suggest_float("static_dropout",
                                              *_lh(space["static_dropout"])),
        "weight_decay": trial.suggest_float("weight_decay",
                                            *_lh(space["weight_decay"]), log=True),
        "batch_size": trial.suggest_int("batch_size",
                                        *_lh(space["batch_size"]), log=True),
    }
    if model == "lstm":
        p["hidden"] = trial.suggest_int("hidden", *_lh(space["hidden"]))
        p["num_layers"] = trial.suggest_int("num_layers", *_lh(space["num_layers"]))
    else:
        p["nheads"] = trial.suggest_int("nheads", *_lh(space["nheads"]))
        p["head_dim"] = trial.suggest_int("head_dim", *_lh(space["head_dim"]))
        p["nblock"] = trial.suggest_int("nblock", *_lh(space["nblock"]))
    return resolve_params(p, model)


def resolve_params(params: dict, model: str) -> dict:
    """Derive dependent values (transformer hidden) from raw study params.
    Also used by wp5_train_final on params loaded back from the study."""
    p = dict(params)
    if model != "lstm" and "hidden" not in p:
        p["hidden"] = int(p["nheads"]) * int(p["head_dim"])
    return p


def _lh(entry: dict) -> tuple:
    return entry["low"], entry["high"]
