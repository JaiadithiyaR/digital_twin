"""Unit tests for the central configuration loader (src/common/config.py)."""

from __future__ import annotations

from src.common.config import load_secrets, load_settings


def test_settings_load_and_validate():
    settings = load_settings()
    assert settings.environment in ("demo", "live")
    assert settings.telemetry.source in ("mock", "zmq")
    assert settings.ppo.action_mapping == {0: "recalibrate", 1: "regenerate", 2: "expand_scope"}
    assert settings.fidelity.epsilon > 0
    assert settings.fidelity.rolling_window_length > 0
    assert isinstance(settings.dt_models.enable_prb_model, bool)


def test_settings_is_frozen():
    settings = load_settings()
    try:
        settings.environment = "live"  # type: ignore[misc]
        assert False, "Settings should be immutable"
    except Exception:
        pass


def test_settings_path_resolution_is_repo_relative():
    settings = load_settings()
    resolved = settings.resolve_path(settings.storage.d1_current_state_path)
    assert resolved.is_absolute()
    assert str(resolved).endswith("data/artifacts/d1_current_state.parquet")


def test_secrets_never_expose_key_in_repr():
    secrets = load_secrets()
    assert "sk-" not in repr(secrets)
    assert "<redacted>" in repr(secrets)


def test_secrets_require_key_raises_when_absent(monkeypatch):
    from src.common import config as config_module

    monkeypatch.setattr(config_module, "load_secrets", config_module.load_secrets)
    secrets = config_module.Secrets(anthropic_api_key=None)
    try:
        secrets.require_anthropic_key()
        assert False, "Expected RuntimeError when ANTHROPIC_API_KEY is unset"
    except RuntimeError:
        pass
