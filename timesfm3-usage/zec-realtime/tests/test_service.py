"""Tests for the poll loop, driven by a fake OKX client (no network, no weights)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from zecfm import config as cfg
from zecfm.features import FeatureWindow
from zecfm.forecaster import ConfigForecast, Engine, RandomWalkEngine, build_engine
from zecfm.okx import Candle, MarketSnapshot, Sample
from zecfm.service import ForecastService, merge_funding
from zecfm.store import Store


class FakeClient:
  """Replays a deterministic market that advances one bar per poll."""

  def __init__(self, bars: int = cfg.CONTEXT_BARS + cfg.WARMUP_BARS):
    self.base_ts = 1_700_000_000_000
    self.bars = bars
    self.polls = 0

  async def __aenter__(self):
    return self

  async def __aexit__(self, *exc):
    return None

  def _candles(self) -> list[Candle]:
    out = []
    for i in range(self.bars + self.polls):
      price = 100.0 + math.sin(i / 7.0) * 2.0
      out.append(
        Candle(
          ts=self.base_ts + i * cfg.BAR_MS,
          open=price, high=price * 1.002, low=price * 0.998,
          close=price, volume_quote=1_000_000.0,
        )
      )
    return out

  async def fetch_snapshot(self) -> MarketSnapshot:
    candles = self._candles()
    now = candles[-1].ts + cfg.BAR_MS
    stats = [Sample(c.ts, 50.0 + (c.ts % 7)) for c in candles]
    return MarketSnapshot(
      fetched_ms=now,
      candles=candles,
      taker_buy_share=stats,
      long_short_account_ratio=stats,
      margin_loan_ratio=stats,
      open_interest=stats,
      funding_now_bp=-2.5,
      funding_now_ts=now,
      funding_history_bp=[Sample(candles[0].ts, -2.0)],
    )

  async def fetch_ticker(self):
    candles = self._candles()
    return {"last": candles[-1].close, "open_24h": 100.0, "high_24h": 102.0,
            "low_24h": 98.0, "vol_24h_quote": 1.0, "ts": candles[-1].ts}

  def advance(self):
    self.polls += 1


class CountingEngine(Engine):
  name = "counting"
  is_foundation_model = False

  def __init__(self):
    self.calls = 0

  def forecast(self, window: FeatureWindow, config, horizon_bars: int) -> ConfigForecast:
    self.calls += 1
    median = np.full(horizon_bars, window.last_close)
    return ConfigForecast(
      config_id=config.id,
      origin_ts=window.origin_ts,
      horizon_bars=horizon_bars,
      target_ts=self.target_timestamps(window.origin_ts, horizon_bars),
      median=median,
      quantiles=np.repeat(median[:, None], 9, axis=1),
      latency_ms=1.0,
    )


@pytest.fixture()
def service(tmp_path):
  store = Store(str(tmp_path / "svc.db"))
  svc = ForecastService(store, CountingEngine(), horizon_bars=4)
  yield svc
  store.close()


def test_merge_funding_prefers_live_samples_over_settled_history():
  merged = merge_funding(
    [Sample(1000, 1.0), Sample(9000, 2.0)],
    [Sample(9000, 2.5), Sample(12000, 3.0)],
  )
  assert [(s.ts, s.value) for s in merged] == [(1000, 1.0), (9000, 2.5), (12000, 3.0)]


def test_poll_forecasts_once_per_bar(service):
  import asyncio

  client = FakeClient()

  async def run():
    await service.poll_once(client)          # first bar -> one forecast per config
    await service.poll_once(client)          # same bar  -> no new work
    client.advance()
    await service.poll_once(client)          # new bar   -> forecast again

  asyncio.run(run())
  assert service.engine.calls == 2 * len(cfg.CONFIGS)
  for config in cfg.CONFIGS:
    assert service.store.forecast_count(config.id) == 2


def test_poll_persists_bars_and_a_channel_window(service):
  import asyncio

  asyncio.run(service.poll_once(FakeClient()))
  assert len(service.store.recent_bars(1000)) > cfg.CONTEXT_BARS
  window = service.store.latest_channel_window()
  assert window is not None
  assert set(window["channels"]) == set(cfg.ALL_KEYS)
  assert len(window["grid_ts"]) == cfg.CONTEXT_BARS


def test_metrics_cover_every_configuration_even_with_no_samples(service):
  import asyncio

  asyncio.run(service.poll_once(FakeClient()))
  metrics = service.metrics()
  assert [m["config_id"] for m in metrics] == [c.id for c in cfg.CONFIGS]
  for m in metrics:
    assert m["overall"]["n"] == 0
    assert len(m["by_step"]) == service.horizon_bars
    assert m["pending"] == service.horizon_bars


def test_forecasts_resolve_once_the_market_moves_on(service):
  import asyncio

  client = FakeClient()

  async def run():
    await service.poll_once(client)
    for _ in range(6):  # let the market print the bars we predicted
      client.advance()
      await service.poll_once(client)

  asyncio.run(run())
  metrics = {m["config_id"]: m for m in service.metrics()}
  assert metrics["full"]["overall"]["n"] > 0
  assert metrics["full"]["overall"]["mae"] is not None
  assert metrics["full"]["overall"]["skill"] is not None


def test_trails_are_clamped_to_the_configured_horizon(service):
  import asyncio

  asyncio.run(service.poll_once(FakeClient()))
  trails = service.trails(999)
  assert set(trails) == {c.id for c in cfg.CONFIGS}


def test_status_reports_the_documented_window(service):
  status = service.status()
  assert status["context_bars"] == cfg.CONTEXT_BARS
  assert status["context_minutes"] == 355
  assert status["context_hours"] == pytest.approx(5.917, abs=0.001)


# --- engines ----------------------------------------------------------------


def test_random_walk_engine_is_flagged_as_not_a_foundation_model():
  engine = build_engine("random-walk")
  assert isinstance(engine, RandomWalkEngine)
  assert engine.is_foundation_model is False


def test_unknown_engine_is_rejected_rather_than_silently_downgraded():
  with pytest.raises(ValueError, match="unknown engine"):
    build_engine("definitely-not-an-engine")


def test_target_timestamps_start_one_bar_after_the_origin():
  ts = Engine.target_timestamps(1_000_000, 3)
  assert ts.tolist() == [1_000_000 + cfg.BAR_MS * i for i in (1, 2, 3)]
