"""Integration tests for the CLI interface."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Help / Version
# ---------------------------------------------------------------------------


def test_help_output():
    """--help displays usage info containing "convert"."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "convert" in result.stdout.lower()


def test_version_output():
    """--version outputs the version string from pyproject.toml."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    # Version is defined in pyproject.toml as "0.1.0"
    assert "0.1.0" in result.stdout


def test_schema_output():
    """--schema outputs "1.0"."""
    result = runner.invoke(app, ["--schema"])
    assert result.exit_code == 0
    assert "1.0" in result.stdout.strip()


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


def test_basic_convert(fixture_path, tmp_path):
    """Basic convert produces a valid YAML output file."""
    input_file = fixture_path / "novels" / "basic_3ch.txt"
    output_file = tmp_path / "output.yaml"

    result = runner.invoke(app, [
        str(input_file),
        "-o", str(output_file),
    ])

    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    assert output_file.exists(), "Output file was not created"

    with open(output_file, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    assert data is not None
    assert "schema_version" in data
    assert data["schema_version"] == "1.0"
    assert "scenes" in data
    assert "characters" in data


def test_no_cache_flag(fixture_path, tmp_path):
    """--no-cache flag is accepted and pipeline runs."""
    input_file = fixture_path / "novels" / "basic_3ch.txt"
    output_file = tmp_path / "output_nocache.yaml"

    result = runner.invoke(app, [
        str(input_file),
        "-o", str(output_file),
        "--no-cache",
    ])

    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert data is not None
    assert "scenes" in data


def test_confidence_threshold_flag(fixture_path, tmp_path):
    """--confidence-threshold 0.5 is accepted."""
    input_file = fixture_path / "novels" / "basic_3ch.txt"
    output_file = tmp_path / "output_conf.yaml"

    result = runner.invoke(app, [
        str(input_file),
        "-o", str(output_file),
        "--confidence-threshold", "0.5",
    ])

    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    assert output_file.exists()


def test_verbose_flag(fixture_path, tmp_path):
    """--verbose flag is accepted."""
    input_file = fixture_path / "novels" / "basic_3ch.txt"
    output_file = tmp_path / "output_verbose.yaml"

    result = runner.invoke(app, [
        str(input_file),
        "-o", str(output_file),
        "--verbose",
    ])

    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    assert output_file.exists()


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_nonexistent_input_file(tmp_path):
    """Error on nonexistent input file."""
    output_file = tmp_path / "output.yaml"

    result = runner.invoke(app, [
        str(tmp_path / "nonexistent.txt"),
        "-o", str(output_file),
    ])

    assert result.exit_code != 0


def test_invalid_resume_from(fixture_path, tmp_path):
    """--resume-from with invalid stage name gives error."""
    input_file = fixture_path / "novels" / "basic_3ch.txt"
    output_file = tmp_path / "output.yaml"

    result = runner.invoke(app, [
        str(input_file),
        "-o", str(output_file),
        "--resume-from", "invalid_stage",
    ])

    assert result.exit_code != 0
