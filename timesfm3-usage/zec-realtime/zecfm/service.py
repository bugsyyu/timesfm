"""Background poll -> forecast -> score loop, plus the state the dashboard reads.

Cadence: market data is refreshed every `poll_seconds` so the header and the
channel panel stay live, but a *new* forecast is only produced when a new 5m bar
closes. One origin per bar, three configurations per origin, forever.

There is deliberately no backfill. To forecast at some past origin T the taker
buy/sell channel would need 5.9 hours of history ending at T, and the endpoint
only retains 5.9 hours ending *now* -- so the earliest origin config C can be
built for is the present one. Backfilling A and B alone would hand them hundreds
of scored points while C had none, which would make the comparison table
meaningless. All three configurations therefore start empty and fill in together
as the service runs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from . import config as cfg
from .features import FeatureWindow, build_feature_window
from .forecaster import Engine
from .okx import MarketSnapshot, OkxClient, Sample
from .store import Store, score

_LOG = logging.getLogger(__name__)


def merge_funding(
  history: list[Sample], live_samples: list[Sample]
) -> list[Sample]:
  """Combines settled funding history with locally recorded live samples.

  Live samples are bucketed to the 5m grid and win over the 8-hourly settlement
  steps, so once the service has been up for a while the channel carries real
  intra-period movement instead of a flat line.
  """
  merged: dict[int, float] = {s.ts: s.value for s in history}
  merged.update({s.ts: s.value for s in live_samples})
  return [Sample(ts=ts, value=value) for ts, value in sorted(merged.items())]


class ForecastService:
  """Owns the polling loop and the in-memory view of the latest cycle."""

  def __init__(
    self,
    store: Store,
    engine: Engine,
    *,
    poll_seconds: int = cfg.POLL_SECONDS,
    horizon_bars: int = cfg.HORIZON_BARS,
    context_bars: int = cfg.CONTEXT_BARS,
    client: OkxClient | None = None,
  ) -> None:
    self.store = store
    self.engine = engine
    self.poll_seconds = poll_seconds
    self.horizon_bars = horizon_bars
    self.context_bars = context_bars
    self._client = client
    self._task: asyncio.Task[None] | None = None
    self._stopping = asyncio.Event()

    self.started_ms = int(time.time() * 1000)
    self.last_poll_ms: int | None = None
    self.last_forecast_ms: int | None = None
    self.last_origin_ts: int | None = None
    self.last_error: str | None = None
    self.consecutive_errors = 0
    self.poll_count = 0
    self.ticker: dict[str, float] | None = None
    self.coverage: dict[str, float] = {}
    self.latencies: dict[str, float] = {}

  # --- Lifecycle ------------------------------------------------------------

  async def start(self) -> None:
    self._stopping.clear()
    self._task = asyncio.create_task(self._run(), name="zecfm-poller")

  async def stop(self) -> None:
    self._stopping.set()
    if self._task is not None:
      self._task.cancel()
      try:
        await self._task
      except asyncio.CancelledError:
        pass
      self._task = None

  async def _run(self) -> None:
    async with (self._client or OkxClient()) as client:
      while not self._stopping.is_set():
        try:
          await self.poll_once(client)
          self.consecutive_errors = 0
          self.last_error = None
        except asyncio.CancelledError:
          raise
        except Exception as exc:  # keep the loop alive across API hiccups
          self.consecutive_errors += 1
          self.last_error = f"{type(exc).__name__}: {exc}"
          _LOG.exception("poll failed (%d consecutive)", self.consecutive_errors)

        # Back off on repeated failures so a venue outage does not hammer OKX.
        delay = self.poll_seconds * min(2**self.consecutive_errors, 8)
        try:
          await asyncio.wait_for(self._stopping.wait(), timeout=delay)
        except asyncio.TimeoutError:
          pass

  # --- One cycle ------------------------------------------------------------

  async def poll_once(self, client: OkxClient) -> FeatureWindow | None:
    """Refreshes data, and forecasts if a new bar has closed."""
    snapshot = await client.fetch_snapshot()
    try:
      self.ticker = await client.fetch_ticker()
    except Exception as exc:  # the ticker is cosmetic; never fail a cycle on it
      _LOG.warning("ticker fetch failed: %s", exc)

    self.store.upsert_bars(snapshot.candles)
    if snapshot.funding_now_bp is not None and snapshot.funding_now_ts is not None:
      self.store.record_funding(snapshot.funding_now_ts, snapshot.funding_now_bp)

    window = self._build_window(snapshot)
    self.coverage = window.coverage
    self.last_poll_ms = int(time.time() * 1000)
    self.poll_count += 1

    if window.origin_ts != self.last_origin_ts:
      await self._forecast_origin(window)
    return window

  def _build_window(self, snapshot: MarketSnapshot) -> FeatureWindow:
    lookback_start = snapshot.fetched_ms - int(
      (self.context_bars + 2) * cfg.BAR_MS
    )
    funding = merge_funding(
      snapshot.funding_history_bp, self.store.funding_samples(lookback_start)
    )
    return build_feature_window(
      snapshot.candles,
      now_ms=snapshot.fetched_ms,
      taker_buy_share=snapshot.taker_buy_share,
      long_short_account_ratio=snapshot.long_short_account_ratio,
      margin_loan_ratio=snapshot.margin_loan_ratio,
      open_interest=snapshot.open_interest,
      funding_bp=funding,
      context_bars=self.context_bars,
    )

  async def _forecast_origin(self, window: FeatureWindow) -> None:
    """Runs all three configurations for a freshly closed bar."""
    if all(
      self.store.has_forecast(window.origin_ts, c.id) for c in cfg.CONFIGS
    ):
      self.last_origin_ts = window.origin_ts
      return

    # Inference is synchronous CPU/GPU work; keep it off the event loop so the
    # HTTP endpoints stay responsive while the model runs.
    forecasts = await asyncio.to_thread(
      self.engine.forecast_all, window, self.horizon_bars, cfg.CONFIGS
    )

    created_ms = int(time.time() * 1000)
    for forecast in forecasts:
      self.store.save_forecast(
        forecast,
        engine=self.engine.name,
        origin_close=window.last_close,
        context_bars=self.context_bars,
        created_ms=created_ms,
      )
      self.latencies[forecast.config_id] = forecast.latency_ms

    self.store.save_channel_window(window)
    self.last_origin_ts = window.origin_ts
    self.last_forecast_ms = created_ms
    _LOG.info(
      "forecast origin=%s close=%.2f configs=%d",
      window.origin_ts,
      window.last_close,
      len(forecasts),
    )

  # --- State for the API ----------------------------------------------------

  def status(self) -> dict[str, Any]:
    return {
      "engine": self.engine.name,
      "is_foundation_model": self.engine.is_foundation_model,
      "device": getattr(self.engine, "device", None),
      "checkpoint": getattr(self.engine, "checkpoint_path", None),
      "inst_id": cfg.INST_ID,
      "bar": cfg.BAR,
      "context_bars": self.context_bars,
      "context_minutes": self.context_bars * cfg.BAR_MS // 60000,
      "context_hours": round(self.context_bars * cfg.BAR_MS / 3_600_000, 3),
      "horizon_bars": self.horizon_bars,
      "horizon_minutes": self.horizon_bars * cfg.BAR_MS // 60000,
      "poll_seconds": self.poll_seconds,
      "started_ms": self.started_ms,
      "uptime_seconds": int(time.time() - self.started_ms / 1000),
      "last_poll_ms": self.last_poll_ms,
      "last_forecast_ms": self.last_forecast_ms,
      "last_origin_ts": self.last_origin_ts,
      "poll_count": self.poll_count,
      "last_error": self.last_error,
      "consecutive_errors": self.consecutive_errors,
      "latency_ms": {k: round(v, 1) for k, v in self.latencies.items()},
      "coverage": {k: round(v, 4) for k, v in self.coverage.items()},
      "server_ms": int(time.time() * 1000),
    }

  def metrics(self) -> list[dict[str, Any]]:
    """Per-configuration accuracy, overall and broken down by lead time."""
    out: list[dict[str, Any]] = []
    for config in cfg.CONFIGS:
      points = self.store.resolved_points(config.id)
      overall = score(points)
      by_step = []
      for step in range(1, self.horizon_bars + 1):
        step_points = [p for p in points if p["step"] == step]
        stats = score(step_points)
        by_step.append(
          {
            "step": step,
            "minutes": step * cfg.BAR_MS // 60000,
            "mae": stats["mae"],
            "mape": stats["mape"],
            "direction_acc": stats["direction_acc"],
            "skill": stats["skill"],
            "n": stats["n"],
          }
        )
      out.append(
        {
          "config_id": config.id,
          "name_zh": config.name_zh,
          "name_en": config.name_en,
          "color": config.color,
          "blurb_zh": config.blurb_zh,
          "num_covariates": config.num_covariates,
          "channels": list(config.channels),
          "forecasts": self.store.forecast_count(config.id),
          "pending": self.store.pending_count(config.id),
          "overall": overall,
          "by_step": by_step,
        }
      )
    return out

  def trails(self, step: int) -> dict[str, list[dict[str, Any]]]:
    """Fixed-lead-time prediction tracks, for the predicted-vs-actual overlay.

    Holding the lead time constant (say "every 60-minutes-ahead prediction we
    ever made") turns a pile of overlapping forecasts into one continuous line
    that can be laid directly over the realized price.
    """
    step = max(1, min(step, self.horizon_bars))
    return {
      config.id: self.store.resolved_points(config.id, step=step, limit=400)
      for config in cfg.CONFIGS
    }

  def live_forecasts(self) -> dict[str, Any]:
    return {config.id: self.store.latest_forecast(config.id) for config in cfg.CONFIGS}
