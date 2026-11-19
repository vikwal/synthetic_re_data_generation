"""Login and per-user separation. Run: pytest fastapi_app/tests/test_auth.py"""
import pytest

from fastapi_app import auth, settings


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "SESSION_HOURS", 1.0)
    monkeypatch.delenv("SYNTH_RE_SECRET", raising=False)
    auth._failures.clear()
    return tmp_path


def test_hash_roundtrip():
    h = auth.hash_password("correct horse battery")
    assert auth.verify_password("correct horse battery", h)
    assert not auth.verify_password("wrong", h)
    assert not auth.verify_password("x", "garbage")


def test_add_user_rules(store):
    auth.add_user("Alice", "longenoughpw")
    assert "alice" in auth.list_users()
    with pytest.raises(auth.AuthError):
        auth.add_user("alice", "longenoughpw")          # duplicate
    with pytest.raises(auth.AuthError):
        auth.add_user("bob", "short")                   # too short
    with pytest.raises(auth.AuthError):
        auth.add_user("../x", "longenoughpw")           # bad name


def test_login_and_token(store):
    auth.add_user("alice", "longenoughpw")
    tok = auth.login("alice", "longenoughpw", "1.1.1.1")
    assert auth.user_from_token(tok) == "alice"
    with pytest.raises(auth.AuthError):
        auth.login("alice", "nope", "1.1.1.1")
    with pytest.raises(auth.AuthError):
        auth.login("ghost", "longenoughpw", "1.1.1.1")


def test_token_tamper_expiry_and_password_change(store):
    auth.add_user("alice", "longenoughpw")
    tok = auth.login("alice", "longenoughpw")
    assert auth.user_from_token(tok + "0") is None
    assert auth.user_from_token("a." + tok.split(".")[1]) is None
    import time
    assert auth.user_from_token(tok, now=time.time() + 2 * 3600) is None          # expired
    auth.set_password("alice", "anotherlongpw")
    assert auth.user_from_token(tok) is None                                     # old session ended
    auth.delete_user("alice")
    assert auth.user_from_token(auth.make_token("alice", "x")) is None


def test_throttle(store):
    auth.add_user("alice", "longenoughpw")
    for _ in range(auth.MAX_FAILURES):
        with pytest.raises(auth.AuthError):
            auth.login("alice", "bad", "9.9.9.9")
    with pytest.raises(auth.AuthError) as e:
        auth.login("alice", "longenoughpw", "9.9.9.9")
    assert e.value.status == 429


def test_storage_is_separate_per_user(store):
    pytest.importorskip("pandas")
    from fastapi_app import storage
    with pytest.raises(auth.AuthError):
        storage.list_scenarios()                                                 # nobody signed in
    t = auth.set_current("alice")
    try:
        storage.create_scenario("A scenario", "wind", {})
        assert len(storage.list_scenarios()) == 1
    finally:
        auth.reset_current(t)
    t = auth.set_current("bob")
    try:
        assert storage.list_scenarios() == []
    finally:
        auth.reset_current(t)
    assert (store / "users" / "alice").exists()


def test_adopt_legacy(store):
    from fastapi_app import manage_users
    (store / "scenarios").mkdir()
    (store / "scenarios" / "s1.json").write_text("{}")
    auth.add_user("alice", "longenoughpw")
    assert manage_users.adopt_legacy("alice") == ["scenarios"]
    assert (store / "users" / "alice" / "scenarios" / "s1.json").exists()
    assert not (store / "scenarios").exists()
