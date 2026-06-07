"""FastAPI web server for the Novel-to-Script Adaptation pipeline.

Provides a REST + SSE endpoint for the browser frontend and serves
static files from the ``static/`` directory.

Usage::

    novel2script-web          # after pip install
    python -m server.app      # direct invocation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Suppress jieba's noisy DEBUG log
logging.getLogger("jieba").setLevel(logging.WARNING)

app = FastAPI(title="Novel-to-Script", version="0.2.0")


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class ConvertRequest(BaseModel):
    text: str
    use_ai: bool = False


# ---------------------------------------------------------------------------
# Pipeline runner (sync, runs in executor)
# ---------------------------------------------------------------------------


def _run_pipeline(
    text: str,
    use_ai: bool,
    progress_callback: callable[[str], None] | None,
) -> dict:
    """Execute the full pipeline on *text* and return YAML + stats.

    Called inside ``loop.run_in_executor`` so it doesn't block the
    asyncio event loop.
    """
    from engine.converter import Pipeline

    import yaml as _yaml

    # Write text to a temp file (Pipeline requires a file path for caching).
    # TemporaryDirectory auto-cleans on exit, even if the process is killed.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "input.txt")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)

        result = Pipeline().run(
            input_path=tmp_path,
            use_ai=use_ai,
            progress_callback=progress_callback,
        )

        # Patch source_novel to a friendly name instead of the temp path
        result.source_novel = "网页输入"

        # Count chapters from unique chapter IDs in scenes
        chapter_ids = set(sc.chapter_id for sc in result.scenes)

        yaml_str = _yaml.dump(
            result.model_dump(mode="python"),
            allow_unicode=True,
            sort_keys=False,
        )

        return {
            "yaml": yaml_str,
            "stats": {
                "scenes": len(result.scenes),
                "characters": len(result.characters),
                "chapters": len(chapter_ids),
                "title": result.title,
            },
        }


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------


@app.post("/api/convert")
async def convert(req: ConvertRequest):
    """Convert novel text and stream progress + result via SSE.

    Events
    ------
    ``event: stage``
        Pipeline stage started.
        ``data: {"stage": "分章"}``

    ``event: result``
        Final YAML output + stats.
        ``data: {"yaml": "...", "stats": {...}}``

    ``event: done``
        Stream end marker.
    """
    if not req.text.strip():
        return StreamingResponse(
            _error_stream("输入文本为空"),
            media_type="text/event-stream",
        )

    queue: asyncio.Queue = asyncio.Queue()

    def progress_callback(stage: str) -> None:
        """Sync callback — pushes stages into the async queue."""
        try:
            queue.put_nowait(("stage", stage))
        except asyncio.QueueFull:
            pass

    async def event_generator():
        loop = asyncio.get_event_loop()

        # Run the (synchronous, potentially long-running) pipeline in a
        # thread-pool executor so we can stream progress concurrently.
        future = loop.run_in_executor(
            None,
            _run_pipeline,
            req.text,
            req.use_ai,
            progress_callback,
        )

        # Drain the queue until the pipeline future is done, then send
        # the final result.
        while True:
            try:
                event_type, data = await asyncio.wait_for(
                    queue.get(), timeout=0.15
                )
                if event_type == "stage":
                    yield _sse("stage", {"stage": data})
            except asyncio.TimeoutError:
                if future.done():
                    break

        # Retrieve result and send as final SSE event
        try:
            result = future.result()
            yield _sse("result", result)
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

        yield _sse("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    """Format a dict as an SSE event pair."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _error_stream(message: str):
    """Yield a single-error SSE stream."""
    yield _sse("error", {"message": message})
    yield _sse("done", {})


# ---------------------------------------------------------------------------
# Static file serving (must be last — after all routes are registered)
# ---------------------------------------------------------------------------

_static_dir = Path(__file__).parent.parent.parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Start the uvicorn server."""
    import webbrowser

    import uvicorn

    host = os.environ.get("NOVEL2SCRIPT_HOST", "0.0.0.0")
    port = int(os.environ.get("NOVEL2SCRIPT_PORT", "8000"))

    # Open browser after a short delay (uvicorn startup is async)
    def _open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(f"http://{('127.0.0.1' if host == '0.0.0.0' else host)}:{port}")

    import threading
    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run("server.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
