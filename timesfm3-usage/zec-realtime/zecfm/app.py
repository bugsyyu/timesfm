"""FastAPI application: one JSON endpoint the dashboard polls, plus the page."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config as cfg
from .forecaster import build_engine
from .service import ForecastService
from .store import Store

_LOG = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def definition_payload() -> dict[str, Any]:
  """The static experiment definition: channels and configurations."""
  return {
    "inst_id": cfg.INST_ID,
    "bar": cfg.BAR,
    "context_bars": cfg.CONTEXT_BARS,
    "context_minutes": cfg.CONTEXT_BARS * cfg.BAR_MS // 60000,
    "context_hours": round(cfg.CONTEXT_HOURS, 2),
    "horizon_bars": cfg.HORIZON_BARS,
    "quantile_levels": list(cfg.QUANTILE_LEVELS),
    "channels": [dataclasses.asdict(c) for c in cfg.ALL_CHANNELS],
    "configs": [dataclasses.asdict(c) for c in cfg.CONFIGS],
  }


def create_app(
  *,
  db_path: str | None = None,
  engine_kind: str | None = None,
  checkpoint_path: str | None = None,
  device: str | None = None,
  horizon_bars: int | None = None,
  poll_seconds: int | None = None,
  autostart: bool = True,
) -> FastAPI:
  """Builds the app. Settings fall back to environment variables so the module
  works both under `python -m zecfm` and under an external ASGI server."""

  db_path = db_path or os.environ.get("ZECFM_DB", "zecfm.db")
  engine_kind = engine_kind or os.environ.get("ZECFM_ENGINE", "timesfm")
  checkpoint_path = checkpoint_path or os.environ.get(
    "ZECFM_CHECKPOINT", "google/timesfm-3.0-pytorch"
  )
  device = device or os.environ.get("ZECFM_DEVICE") or None
  horizon_bars = horizon_bars or int(
    os.environ.get("ZECFM_HORIZON", cfg.HORIZON_BARS)
  )
  poll_seconds = poll_seconds or int(os.environ.get("ZECFM_POLL", cfg.POLL_SECONDS))

  state: dict[str, Any] = {}

  @contextlib.asynccontextmanager
  async def lifespan(app: FastAPI):
    store = Store(db_path)
    engine = build_engine(
      engine_kind, checkpoint_path=checkpoint_path, device=device
    )
    service = ForecastService(
      store,
      engine,
      poll_seconds=poll_seconds,
      horizon_bars=horizon_bars,
    )
    state["store"] = store
    state["service"] = service
    if autostart:
      await service.start()
    try:
      yield
    finally:
      await service.stop()
      store.close()

  app = FastAPI(
    title="ZEC-USDT realtime forecasting with TimesFM 3.0",
    version="1.0.0",
    lifespan=lifespan,
  )

  def service() -> ForecastService:
    return state["service"]

  @app.get("/api/definition")
  def get_definition() -> dict[str, Any]:
    return definition_payload()

  @app.get("/api/status")
  def get_status() -> dict[str, Any]:
    return service().status()

  @app.get("/api/metrics")
  def get_metrics() -> list[dict[str, Any]]:
    return service().metrics()

  @app.get("/api/state")
  def get_state(
    bars: int = Query(96, ge=12, le=1000, description="Closed bars of realized price"),
    lead: int = Query(
      cfg.HORIZON_BARS, ge=1, description="Lead time, in bars, for the accuracy track"
    ),
  ) -> dict[str, Any]:
    svc = service()
    return {
      "definition": definition_payload(),
      "status": svc.status(),
      "ticker": svc.ticker,
      "bars": svc.store.recent_bars(bars),
      "live": svc.live_forecasts(),
      "metrics": svc.metrics(),
      "trails": svc.trails(lead),
      "lead": max(1, min(lead, svc.horizon_bars)),
      "channels": svc.store.latest_channel_window(),
    }

  @app.get("/api/healthz")
  def healthz() -> JSONResponse:
    svc = service()
    healthy = svc.consecutive_errors < 5
    return JSONResponse(
      {
        "ok": healthy,
        "last_poll_ms": svc.last_poll_ms,
        "last_error": svc.last_error,
        "engine": svc.engine.name,
      },
      status_code=200 if healthy else 503,
    )

  @app.get("/")
  def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

  app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
  return app
