import json

import pandas as pd

from utils import overrides as OVR


def test_latest_entry_wins_and_types(tmp_path):
    p = tmp_path / "ov.json"
    p.write_text(json.dumps([
        {"park_id": "A", "field": "area_ha", "old": 1.0, "new": 2.0, "timestamp": "t1", "comment": ""},
        {"park_id": "A", "field": "area_ha", "old": 2.0, "new": 3.5, "timestamp": "t2", "comment": ""},
        {"park_id": "B", "field": "orientation_class", "old": "Süd", "new": "Ost-West", "timestamp": "t3", "comment": ""},
        {"park_id": "Z", "field": "area_ha", "old": 1.0, "new": 9.0, "timestamp": "t4", "comment": ""},
    ]))
    df = pd.DataFrame({"park_id": ["A", "B"], "area_ha": [1.0, 4.0], "orientation_class": ["Süd", "Süd"]})
    out, applied = OVR.apply(df, str(p), "pv", log=lambda *_: None)
    assert out.area_ha.tolist() == [3.5, 4.0]
    assert out.orientation_class.tolist() == ["Süd", "Ost-West"]
    assert len(applied) == 2                                       # Z is not in the table
    assert df.area_ha.tolist() == [1.0, 4.0]                       # the input stays untouched


def test_from_env_without_variable(monkeypatch):
    monkeypatch.delenv("WIND_OVERRIDES", raising=False)
    df = pd.DataFrame({"turbine_id": ["T1"], "hub_height": [100.0]})
    assert OVR.from_env(df, "wind") is df
