"""Blocking HTTP client for a remote ``SelfImprovementMathEnv`` server.

Purpose: let an agent (our PPO trainer, a TRL ``GRPOTrainer``, a demo
notebook, or a reviewer with curl) drive a running server with the
same ergonomic shape as the in-process ``SelfImprovementMathEnv``.

We intentionally use plain ``requests`` (synchronous, one-episode-at-a-
time) rather than aiohttp.  Rationale: the env is a single-step
episode + heavy per-step cost (triple-verifier generation), so
batching at the HTTP layer buys little but complicates agent code.  If
you need throughput later, run multiple clients in parallel - the
FastAPI server handles that with the default uvicorn worker pool.
"""

from __future__ import annotations

from typing import Optional

import requests

from src.openenv.models import (
    Action,
    Observation,
    ResetRequest,
    StateResponse,
    StepRequest,
    StepResponse,
)


class SelfImprovementMathClient:
    """Thin HTTP client that mirrors ``SelfImprovementMathEnv`` methods."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Health / metadata
    # ------------------------------------------------------------------
    def health(self) -> dict:
        r = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def metadata(self) -> dict:
        r = self._session.get(f"{self.base_url}/metadata", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # OpenEnv contract
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        requested_topic: Optional[str] = None,
    ) -> Observation:
        payload = ResetRequest(seed=seed, requested_topic=requested_topic).model_dump()
        r = self._session.post(
            f"{self.base_url}/reset", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return Observation.model_validate(r.json())

    def step(self, action: Action) -> StepResponse:
        payload = StepRequest(action=action).model_dump()
        r = self._session.post(
            f"{self.base_url}/step", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return StepResponse.model_validate(r.json())

    def state(self) -> StateResponse:
        r = self._session.get(f"{self.base_url}/state", timeout=self.timeout)
        r.raise_for_status()
        return StateResponse.model_validate(r.json())

    def close(self) -> None:
        """Tell the server the client is done, then release local resources."""
        try:
            self._session.post(f"{self.base_url}/close", timeout=self.timeout)
        finally:
            self._session.close()

    # Context-manager sugar so ``with SelfImprovementMathClient(...) as c:`` works.
    def __enter__(self) -> "SelfImprovementMathClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
