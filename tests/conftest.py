"""Shared test fixtures for the novel-to-script test suite."""

from pathlib import Path
import pytest


@pytest.fixture
def fixture_path():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def load_novel(fixture_path):
    """Return a function that loads a test novel by filename."""

    def _load(name: str) -> str:
        filepath = fixture_path / "novels" / name
        return filepath.read_text(encoding="utf-8")

    return _load


@pytest.fixture
def load_expected(fixture_path):
    """Return a function that loads an expected YAML file by filename."""

    def _load(name: str) -> dict:
        import yaml

        filepath = fixture_path / "expected" / name
        return yaml.safe_load(filepath.read_text(encoding="utf-8"))

    return _load


@pytest.fixture
def basic_novel(load_novel):
    """Return the full text of basic_3ch.txt."""
    return load_novel("basic_3ch.txt")


@pytest.fixture
def mixed_quotes_novel(load_novel):
    """Return the full text of mixed_quotes.txt."""
    return load_novel("mixed_quotes.txt")


@pytest.fixture
def no_dialogue_novel(load_novel):
    """Return the full text of no_dialogue.txt."""
    return load_novel("no_dialogue.txt")
