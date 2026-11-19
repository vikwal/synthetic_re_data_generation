"""External weather service client: request once per location, cache per user. Run: pytest fastapi_app/tests/test_pvservice.py"""
import pytest

from fastapi_app import auth, pvservice, settings


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(settings, "AUTH_ENABLED", False)
    monkeypatch.setattr(settings, "PV_SERVICE_URL", "")
    monkeypatch.setattr(settings, "PV_SERVICE_START", "2023-01-01")
    monkeypatch.setattr(settings, "PV_SERVICE_END", "2023-01-10")
    calls = []
    real = pvservice.request_service

    def counting(lat, lon, start, end):
        calls.append((lat, lon, start, end))
        return real(lat, lon, start, end)

    monkeypatch.setattr(pvservice, "request_service", counting)
    return tmp_path, calls


def test_location_is_snapped():
    assert pvservice.snap(52.3712, 9.7338) == (52.37, 9.73)
    assert pvservice.location_key(52.3712, 9.7338) == pvservice.location_key(52.3749, 9.7301)
    assert pvservice.location_key(-1.5, -2.5) == "latm1.50_lonm2.50"


def test_mock_payload_is_plausible_and_repeatable():
    a = pvservice._mock_payload(52.37, 9.73, "2023-06-01", "2023-06-03")
    b = pvservice._mock_payload(52.37, 9.73, "2023-06-01", "2023-06-03")
    assert a == b                                              # same place, same series
    df, _ = pvservice.parse_payload(a)
    assert len(df) == 72 and df["ghi"].max() > 300 and df["ghi"].min() == 0
    assert (df["dhi"] <= df["ghi"]).all()


def test_parse_rejects_bad_answers():
    with pytest.raises(ValueError):
        pvservice.parse_payload({"timestamps": ["2023-01-01T00:00:00Z"]})
    ok = pvservice._mock_payload(50, 8, "2023-01-01", "2023-01-01")
    bad = dict(ok, ghi=ok["ghi"][:-1])
    with pytest.raises(ValueError):
        pvservice.parse_payload(bad)


def test_second_request_for_same_location_uses_cache(env):
    _, calls = env
    df1, i1 = pvservice.get_location_data(52.3712, 9.7338)
    df2, i2 = pvservice.get_location_data(52.3749, 9.7301)       # same snapped location
    assert len(calls) == 1
    assert i1["cached"] is False and i2["cached"] is True
    assert list(df1.columns) == list(pvservice.COLUMNS) and len(df1) == len(df2) == 240
    pvservice.get_location_data(48.1, 11.6)                       # another place -> new request
    assert len(calls) == 2


def test_changed_period_requests_again(env, monkeypatch):
    _, calls = env
    pvservice.get_location_data(52.37, 9.73)
    monkeypatch.setattr(settings, "PV_SERVICE_END", "2023-01-12")
    df, info = pvservice.get_location_data(52.37, 9.73)
    assert len(calls) == 2 and info["cached"] is False and len(df) == 288


def test_cache_is_per_user(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "PV_SERVICE_START", "2023-01-01")
    monkeypatch.setattr(settings, "PV_SERVICE_END", "2023-01-03")
    n = []
    real = pvservice.request_service
    monkeypatch.setattr(pvservice, "request_service", lambda *a: (n.append(1), real(*a))[1])
    for user in ("alice", "bob", "alice"):
        t = auth.set_current(user)
        try:
            pvservice.get_location_data(52.37, 9.73)
        finally:
            auth.reset_current(t)
    assert len(n) == 2                                            # alice once, bob once, alice again from her cache
    assert (tmp_path / "users" / "alice" / "pvdata").exists() and (tmp_path / "users" / "bob" / "pvdata").exists()


def test_damaged_cache_is_requested_again(env):
    root, calls = env
    pvservice.get_location_data(52.37, 9.73)
    (root / "pvdata" / (pvservice.location_key(52.37, 9.73) + ".csv.gz")).write_bytes(b"junk")
    _, info = pvservice.get_location_data(52.37, 9.73)
    assert info["cached"] is False and len(calls) == 2
