"""Entrypoint discovered by Hugging Face Spaces (Docker SDK).

Spaces conventionally runs ``python app.py`` from the repo root - this
file therefore just defers to ``src.openenv.server`` so we don't
duplicate the FastAPI definition in two places.

It respects the standard HF Space env vars:

* ``PORT`` - default 7860 on Spaces, override locally with ``-p``.
* ``BASE_MODEL``, ``BASE_MODEL_NAME``, ``DEVICE``,
  ``CURRICULUM_CHECKPOINT_DIR`` - see ``src.openenv.server.ServerConfig``.

For local dev you can also run the server directly with::

    python -m src.openenv.server --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os

import uvicorn

from src.openenv.server import app


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    log_level = os.environ.get("LOG_LEVEL", "info")
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=log_level)


if __name__ == "__main__":
    main()
