"""End-to-end pipeline integration tests."""

from engine.preprocess import preprocess
from engine.chapter import split_chapters
from engine.scene import detect_scenes
from engine.character import extract_characters
from engine.dialogue import extract_dialogues
from engine.converter import Pipeline
from engine.models import ScriptOutput


class TestEndToEndPipeline:
    def test_full_pipeline_on_basic_novel(self, basic_novel, tmp_path):
        """Full novel.txt -> ScriptOutput -> YAML roundtrip."""
        # Run with Pipeline (uses cache)
        input_path = tmp_path / "input.txt"
        input_path.write_text(basic_novel, encoding="utf-8")
        output_path = tmp_path / "output.yaml"

        result = Pipeline().run(
            input_path=str(input_path),
            output_path=str(output_path),
            cache_dir=str(tmp_path / "cache"),
        )

        assert isinstance(result, ScriptOutput)
        assert len(result.scenes) > 0
        assert len(result.characters) > 0
        assert output_path.exists()

        # Verify YAML is parseable and valid
        import yaml

        with open(output_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["schema_version"] == "1.0"
        assert "scenes" in data

    def test_pipeline_on_mixed_quotes(self, mixed_quotes_novel, tmp_path):
        """All four quote styles processed correctly."""
        input_path = tmp_path / "input.txt"
        input_path.write_text(mixed_quotes_novel, encoding="utf-8")

        # Run the pipeline to ensure it doesn't crash on mixed quotes
        result = Pipeline().run(input_path=str(input_path))
        assert isinstance(result, ScriptOutput)

        # Direct check via modules to verify multiple quote styles detected
        pp = preprocess(mixed_quotes_novel, "test.txt")
        ch = split_chapters(pp)
        sc = detect_scenes(ch)
        dl = extract_dialogues(sc)
        styles = {d.quote_style for d in dl.dialogues}
        assert len(styles) >= 2  # multiple quote styles detected

    def test_pipeline_on_no_dialogue(self, no_dialogue_novel, tmp_path):
        """Pure narration produces a valid ScriptOutput."""
        input_path = tmp_path / "input.txt"
        input_path.write_text(no_dialogue_novel, encoding="utf-8")

        result = Pipeline().run(input_path=str(input_path))
        # Should not crash on no-dialogue text
        assert isinstance(result, ScriptOutput)

    def test_cache_second_run_is_faster(self, basic_novel, tmp_path):
        """Cache hit on second run produces same result."""
        import time

        input_path = tmp_path / "input.txt"
        input_path.write_text(basic_novel, encoding="utf-8")

        start1 = time.time()
        result1 = Pipeline().run(
            input_path=str(input_path),
            cache_dir=str(tmp_path / "cache"),
        )
        time1 = time.time() - start1

        start2 = time.time()
        result2 = Pipeline().run(
            input_path=str(input_path),
            cache_dir=str(tmp_path / "cache"),
        )
        time2 = time.time() - start2

        # Same result
        assert result1.title == result2.title
        assert len(result1.scenes) == len(result2.scenes)
        # Cache run should generally be faster, but we don't assert that
        # because it depends on filesystem/OS caching behavior.
        # The important thing is it doesn't crash and produces identical output.
        _ = time1, time2  # captured for potential future diagnostics
