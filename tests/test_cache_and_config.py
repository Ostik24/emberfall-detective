"""DiskCache round-trips and the secret lookup order (Colab Secrets, then .env)."""

import os
import sys
import types

import pytest

from concierge import config
from concierge.cache import DiskCache


def test_cache_round_trip_and_key_order(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get({"a": 1, "b": 2}) is None
    cache.put({"a": 1, "b": 2}, {"answer": "Twister"})
    assert cache.get({"b": 2, "a": 1}) == {"answer": "Twister"}
    assert cache.get({"a": 1, "b": 3}) is None


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_SECRETS", {})  # no value remembered from another test
    path = tmp_path / ".env"
    path.write_text("TMDB_API_KEY=from-dotenv\n")
    monkeypatch.setattr(config, "ENV_FILE", path)
    return path


@pytest.fixture
def fake_colab(monkeypatch):
    """Install a stand-in `google.colab.userdata` holding the given secrets."""

    def install(secrets: dict[str, str]) -> None:
        userdata = types.ModuleType("google.colab.userdata")

        class SecretNotFoundError(Exception):
            pass

        def get(name):
            if name not in secrets:
                raise SecretNotFoundError(name)
            return secrets[name]

        userdata.SecretNotFoundError = SecretNotFoundError
        userdata.get = get
        colab = types.ModuleType("google.colab")
        colab.userdata = userdata
        google = types.ModuleType("google")
        google.colab = colab
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.colab", colab)
        monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)

    return install


def test_secret_from_dotenv_outside_colab(env_file, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails, as outside Colab
    assert config.get_secret("TMDB_API_KEY") == "from-dotenv"


def test_colab_secret_wins_over_dotenv(env_file, fake_colab):
    fake_colab({"TMDB_API_KEY": "from-colab"})
    assert config.get_secret("TMDB_API_KEY") == "from-colab"


def test_dotenv_used_when_colab_lacks_the_secret(env_file, fake_colab):
    fake_colab({})
    assert config.get_secret("TMDB_API_KEY") == "from-dotenv"


def test_missing_secret_raises(env_file, fake_colab):
    fake_colab({})
    with pytest.raises(config.MissingSecretError):
        config.get_secret("OPENAI_API_KEY")


def test_secret_lookup_does_not_touch_process_env(env_file, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    config.get_secret("TMDB_API_KEY")
    assert "TMDB_API_KEY" not in os.environ


def test_a_secret_read_once_is_remembered(env_file, fake_colab):
    # A key read once in the main thread is reused later (e.g. by Gradio's worker threads).
    fake_colab({"TMDB_API_KEY": "from-colab"})
    assert config.get_secret("TMDB_API_KEY") == "from-colab"
    fake_colab({})
    env_file.write_text("")
    assert config.get_secret("TMDB_API_KEY") == "from-colab"
