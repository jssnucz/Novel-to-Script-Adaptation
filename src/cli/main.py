"""CLI entry point for the novel-to-script converter.

Usage::

    novel2script INPUT -o OUTPUT
    novel2script INPUT -o OUTPUT --no-cache
    novel2script INPUT -o OUTPUT --resume-from scene
    novel2script INPUT -o OUTPUT --confidence-threshold 0.5
    novel2script INPUT -o OUTPUT --verbose
    novel2script INPUT -o OUTPUT --ai
    novel2script --version
    novel2script --schema
"""

from __future__ import annotations

import logging
import os

import typer

from engine.converter import Pipeline

# Suppress jieba's noisy DEBUG log (prefix dict / model loading messages)
logging.getLogger("jieba").setLevel(logging.WARNING)

app = typer.Typer(
    name="novel2script",
    help="Convert Chinese novels into structured YAML screenplays.",
)


# ---------------------------------------------------------------------------
# Eager callbacks for flags that should exit before argument validation
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    """Print version and exit when --version is passed."""
    if value:
        from importlib.metadata import version as get_version

        typer.echo(get_version("novel-to-script"))
        raise typer.Exit()


def _schema_callback(value: bool) -> None:
    """Print schema version and exit when --schema is passed."""
    if value:
        typer.echo("1.0")
        raise typer.Exit()


@app.command()
def convert(
    input: str = typer.Argument(..., help="Input novel file path (.txt/.md)"),
    output: str = typer.Option(..., "-o", "--output", help="Output YAML script path"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Force re-run all stages"),
    resume_from: str | None = typer.Option(
        None, "--resume-from", help="Resume from a specific stage"
    ),
    confidence_threshold: float = typer.Option(
        0.0,
        "--confidence-threshold",
        help="Filter attributions below this confidence",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Show per-stage timing and stats"
    ),
    use_ai: bool = typer.Option(
        False,
        "--ai",
        help="Enable LLM enhancement (requires NOVEL2SCRIPT_API_KEY env var)",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit",
        is_eager=True,
        callback=_version_callback,
    ),
    schema: bool = typer.Option(
        False,
        "--schema",
        help="Show schema version and exit",
        is_eager=True,
        callback=_schema_callback,
    ),
) -> None:
    """Convert a Chinese novel text file into a structured YAML screenplay."""
    # Warn if --ai is set but no API key configured
    if use_ai and not os.environ.get("NOVEL2SCRIPT_API_KEY"):
        typer.echo(
            "⚠  NOVEL2SCRIPT_API_KEY is not set. "
            "AI enhancement will be skipped.\n"
            "   Set it via:  export NOVEL2SCRIPT_API_KEY=sk-...\n"
            "   Or remove --ai to run with rule-engine only.",
            err=True,
        )

    result = Pipeline().run(
        input_path=input,
        output_path=output,
        no_cache=no_cache,
        resume_from=resume_from,
        confidence_threshold=confidence_threshold,
        verbose=verbose,
        use_ai=use_ai,
    )
    typer.echo(
        f"Converted {len(result.scenes)} scenes with "
        f"{len(result.characters)} characters."
    )
    typer.echo(f"Output → {Path(output).resolve()}")


if __name__ == "__main__":
    app()
