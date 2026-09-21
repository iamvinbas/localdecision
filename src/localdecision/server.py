"""HTTP server (FastAPI).

    POST /v1/systemone   state + typed questions -> typed answers   (System One wire format)
    GET  /v1/models      served model and aliases
    GET  /health         backend, self-test, calibration

Inference runs on one dedicated worker thread: one model is resident, requests are serialized,
and all parallelism happens inside a request (shared prefix + batched views).
"""

from __future__ import annotations

import asyncio
import hmac
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .engine import Engine
from .schema import ModelCard, ModelList, SystemOneRequest, SystemOneResponse

ALIASES = ("localdecision-latest",)


def create_app(engine: Engine, api_key: str | None = None) -> FastAPI:
    api_key = api_key if api_key is not None else os.environ.get("LOCALDECISION_API_KEY")
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="localdecision")
    app = FastAPI(
        title="LocalDecision",
        version=__version__,
        description="Typed, calibrated decisions from a local open-weight model.",
    )

    def authorize(request: Request) -> None:
        if not api_key:
            return
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(token.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.post("/v1/systemone", response_model=SystemOneResponse, response_model_exclude_none=True)
    async def system_one(body: SystemOneRequest, _: None = Depends(authorize)) -> SystemOneResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(worker, engine.run, body)

    @app.get("/v1/models", response_model=ModelList)
    async def models(_: None = Depends(authorize)) -> ModelList:
        today = date.today().isoformat()
        described = f"{engine.backend.model_id} via {engine.backend.name}"
        cards = [
            ModelCard(name=alias, description=f"Alias of {engine.model_name}", release_date=today)
            for alias in ALIASES
        ]
        cards.append(ModelCard(name=engine.model_name, description=described, release_date=today))
        return ModelList(models=cards)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", **engine.describe()}

    return app


def serve(
    engine: Engine, host: str = "127.0.0.1", port: int = 8080, api_key: str | None = None
) -> None:
    import uvicorn

    uvicorn.run(create_app(engine, api_key), host=host, port=port, log_level="info")
