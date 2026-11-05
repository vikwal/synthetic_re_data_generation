"""Pydantic request/response models (Python 3.9 compatible: Optional/List, no `X | None`)."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from . import settings


class TurbineIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    turbine: str = Field(min_length=1, max_length=120)
    hub_height: float = Field(gt=0, le=300)
    commissioning_date: Optional[str] = None


class WindParams(BaseModel):
    turbines: List[TurbineIn] = Field(default_factory=list, max_length=settings.MAX_TURBINES)
    apply_ageing: bool = False
    # Fraction of output lost per year of age. 0.0063 matches config.yaml and the pipeline default.
    annual_degradation: float = Field(default=0.0063, ge=0, le=0.05)
    default_commissioning_date: Optional[str] = "2013-01-01"
    apply_wake: bool = True
    wake_decay: float = Field(default=0.075, ge=0.01, le=0.3)


class SolarParams(BaseModel):
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lon: Optional[float] = Field(default=None, ge=-180, le=180)
    dc_rating_watts: float = Field(default=10000, gt=0)
    surface_tilt: float = Field(default=35, ge=0, le=90)
    surface_azimuth: float = Field(default=180, ge=0, le=360)
    gamma_pdc: float = -0.0035
    albedo: float = Field(default=0.25, ge=0, le=1)
    eta_inv_nom: float = Field(default=0.96, gt=0, le=1)


class ScenarioCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: str

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        if v not in ("wind", "solar"):
            raise ValueError("kind must be 'wind' or 'solar'")
        return v

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be empty")
        return v


class ScenarioUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    params: Optional[dict] = None


class LayoutRequest(BaseModel):
    center_lat: float = Field(ge=-90, le=90)
    center_lon: float = Field(ge=-180, le=180)
    rows: int = Field(ge=1, le=50)
    cols: int = Field(ge=1, le=50)
    spacing_along_d: float = Field(default=7.0, gt=0, le=50)
    spacing_across_d: float = Field(default=5.0, gt=0, le=50)
    rotation_deg: float = Field(default=0.0, ge=-360, le=360)
    turbine: str


class CompareRequest(BaseModel):
    result_ids: List[str] = Field(min_length=1, max_length=6)


class RealCompareRequest(BaseModel):
    result_id: str
    csv_text: str = Field(min_length=1)
    unit: str = "W"


class TrainRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=80)
    architecture: str
    result_ids: List[str] = Field(min_length=1, max_length=6)
    lookback: int = Field(default=24, ge=1, le=168)
    epochs: int = Field(default=30, ge=1, le=300)
    hidden: int = Field(default=64, ge=8, le=256)
    layers: int = Field(default=2, ge=1, le=4)
    dropout: float = Field(default=0.1, ge=0, le=0.6)
    lr: float = Field(default=1e-3, gt=0, le=1)
    batch_size: int = Field(default=128, ge=16, le=2048)
    val_frac: float = Field(default=0.2, ge=0.05, le=0.5)
    ridge_alpha: float = Field(default=1.0, gt=0)
    # Split by date: train on everything up to and including this day (YYYY-MM-DD), validate on the rest.
    # None = split by `val_frac` instead.
    train_from: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    train_until: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    early_stopping: bool = True
    patience: int = Field(default=6, ge=1, le=50)         # epochs without improvement before stopping
    monitor: str = "val_rmse"                              # val_rmse | val_mae | train_loss
    lead_from: int = Field(default=12, ge=0, le=57)           # forecast lead-time window (hours) for package weather
    lead_to: int = Field(default=36, ge=0, le=57)
    refit_full: bool = True                                  # after early stopping, retrain on train + validation for the best epoch
    min_delta: float = Field(default=0.001, ge=0, le=10)   # smallest drop of the monitored value that counts as improvement
    horizon: int = Field(default=1, ge=1, le=72)          # forecast hours predicted at once (1 = a single hour)
    scaling: str = "standard"                              # input scaling: standard | minmax | none
    # Input features chosen by the user (weather column ids plus "time_of_day" / "season"); None = all available.
    features: Optional[List[str]] = None
    # Where the input weather comes from: "era5" (the run's own weather) or an NWP model id ("icon_d2", "ecmwf").
    weather_source: str = "era5"
    # Several forecast models in one run (their inputs are tagged `<model>:<input>` in `features`). Overrides weather_source.
    weather_sources: Optional[List[str]] = None
    # The plant's own past power as an input (on by default). This is the number of hours between the newest power
    # value the model sees and the hour it predicts (1 = predict the next hour from the hour before); 0 = not used.
    power_lag_hours: int = Field(default=1, ge=0, le=48)

    @field_validator("monitor")
    @classmethod
    def _monitor(cls, v: str) -> str:
        if v not in ("val_rmse", "val_mae", "train_loss"):
            raise ValueError("monitor must be val_rmse, val_mae or train_loss")
        return v

    @field_validator("scaling")
    @classmethod
    def _scaling(cls, v: str) -> str:
        if v not in ("standard", "minmax", "none"):
            raise ValueError("scaling must be standard, minmax or none")
        return v

    @field_validator("architecture")
    @classmethod
    def _arch(cls, v: str) -> str:
        if v not in ("linear", "gru", "tcn", "tft"):
            raise ValueError("architecture must be linear, gru, tcn or tft")
        return v


class FinetuneRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=80)
    feature_result_id: str
    csv_text: str = Field(min_length=1)
    filename: Optional[str] = None
    unit: str = "W"
    capacity_w: Optional[float] = Field(default=None, gt=0)
    epochs: int = Field(default=15, ge=1, le=300)
    lr: float = Field(default=3e-4, gt=0, le=1)
    freeze_backbone: bool = False
    train_from: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    train_until: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    ridge_alpha: float = Field(default=10.0, gt=0)
    val_frac: float = Field(default=0.2, ge=0.05, le=0.5)
