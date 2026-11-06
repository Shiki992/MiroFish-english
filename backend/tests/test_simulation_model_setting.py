import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.llm_provider
from app.config import Config

SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")


def load_script(name):
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    return importlib.import_module(name)


def create_with_twitter(_config):
    runner = load_script("run_twitter_simulation").TwitterSimulationRunner
    return runner._create_model(SimpleNamespace(config={}))


def create_with_reddit(_config):
    runner = load_script("run_reddit_simulation").RedditSimulationRunner
    return runner._create_model(SimpleNamespace(config={}))


def create_with_parallel(config):
    return load_script("run_parallel_simulation").create_model(config)


@pytest.fixture
def claude_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("LLM_MODEL_NAME", "claude-opus-5-5")
    monkeypatch.delenv("LLM_SIMULATION_MODEL_NAME", raising=False)
    monkeypatch.setattr(
        app.llm_provider,
        "create_anthropic_camel_model",
        lambda **kwargs: kwargs["model"],
    )
    return monkeypatch


@pytest.mark.parametrize(
    "create", [create_with_twitter, create_with_reddit, create_with_parallel]
)
def test_agents_use_the_simulation_model(claude_env, create):
    claude_env.setenv("LLM_SIMULATION_MODEL_NAME", "claude-haiku-5-5")

    assert create({}) == "claude-haiku-5-5"


@pytest.mark.parametrize(
    "create", [create_with_twitter, create_with_reddit, create_with_parallel]
)
def test_agents_fall_back_to_the_main_model(claude_env, create):
    assert create({}) == "claude-opus-5-5"


def test_validation_rejects_mixed_providers(monkeypatch):
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "claude-opus-5-5")
    monkeypatch.setattr(Config, "LLM_SIMULATION_MODEL_NAME", "gpt-4o-mini")
    assert any("LLM_SIMULATION_MODEL_NAME" in error for error in Config.validate())

    monkeypatch.setattr(Config, "LLM_SIMULATION_MODEL_NAME", "claude-haiku-5-5")
    assert not any("LLM_SIMULATION_MODEL_NAME" in error for error in Config.validate())
