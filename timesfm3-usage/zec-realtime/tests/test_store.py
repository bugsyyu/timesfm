"""Tests for persistence and out-of-sample scoring."""

from __future__ import annotations

import numpy as np
import pytest

from zecfm import config as cfg
from zecfm.forecaster import ConfigForecast
from zecfm.okx import Candle
from zecfm.store import Store, score


@pytest.fixture()
def store(tmp_path):
  s = Store(str(tmp_path / "test.db"))
  yield s
  s.close()


def _forecast(origin_ts: int, config_id: str, medians: list[float]) -> ConfigForecast:
  horizon = len(medians)
  median = np.array(medians, dtype=np.float64)
  # Nine deciles spread symmetrically around the median.
  offsets = np.linspace(-5, 5, 9)
  return ConfigForecast(
    config_id=config_id,
    origin_ts=origin_ts,
    horizon_bars=horizon,
    target_ts=np.array(
      [origin_ts + (i + 1) * cfg.BAR_MS for i in range(horizon)], dtype=np.int64
    ),
    median=median,
    quantiles=median[:, None] + offsets[None, :],
    latency_ms=12.5,
  )


def _bar(ts: int, close: float) -> Candle:
  return Candle(ts=ts, open=close, high=close, low=close, close=close, volume_quote=1.0)


def test_save_forecast_is_idempotent_per_origin_and_config(store):
  fc = _forecast(1_000_000, "full", [100.0, 101.0])
  first = store.save_forecast(fc, engine="t", origin_close=99.0, context_bars=71, created_ms=1)
  second = store.save_forecast(fc, engine="t", origin_close=99.0, context_bars=71, created_ms=2)
  assert first is not None
  assert second is None, "a second write for the same origin must not duplicate points"
  assert store.forecast_count("full") == 1


def test_points_resolve_only_once_their_bar_prints(store):
  origin = 1_000_000
  fc = _forecast(origin, "full", [100.0, 101.0, 102.0])
  store.save_forecast(fc, engine="t", origin_close=99.0, context_bars=71, created_ms=1)

  assert store.resolved_points("full") == []
  assert store.pending_count("full") == 3

  store.upsert_bars([_bar(origin + cfg.BAR_MS, 100.5)])
  resolved = store.resolved_points("full")
  assert len(resolved) == 1
  assert resolved[0]["actual"] == pytest.approx(100.5)
  assert resolved[0]["step"] == 1
  assert store.pending_count("full") == 2


def test_resolved_points_can_be_filtered_to_one_lead_time(store):
  origin = 2_000_000
  store.save_forecast(
    _forecast(origin, "full", [10.0, 20.0, 30.0]),
    engine="t", origin_close=9.0, context_bars=71, created_ms=1,
  )
  store.upsert_bars([_bar(origin + (i + 1) * cfg.BAR_MS, 11.0) for i in range(3)])
  assert len(store.resolved_points("full")) == 3
  step_two = store.resolved_points("full", step=2)
  assert len(step_two) == 1 and step_two[0]["step"] == 2


def test_upsert_bars_overwrites_a_revised_bar(store):
  store.upsert_bars([_bar(500, 1.0)])
  store.upsert_bars([_bar(500, 2.0)])
  bars = store.recent_bars()
  assert len(bars) == 1 and bars[0]["close"] == pytest.approx(2.0)


def test_recent_bars_returns_oldest_first(store):
  store.upsert_bars([_bar(300, 3.0), _bar(100, 1.0), _bar(200, 2.0)])
  assert [b["ts"] for b in store.recent_bars()] == [100, 200, 300]


def test_funding_samples_bucket_to_the_grid(store):
  store.record_funding(cfg.BAR_MS + 1234, -2.5)
  store.record_funding(cfg.BAR_MS + 4321, -2.7)  # same bucket, later observation
  samples = store.funding_samples(0)
  assert len(samples) == 1
  assert samples[0].ts == cfg.BAR_MS
  assert samples[0].value == pytest.approx(-2.7)


# --- scoring ----------------------------------------------------------------


def _pt(median, actual, origin, lo=None, hi=None):
  return {
    "median": median, "actual": actual, "origin_close": origin,
    "q10": lo if lo is not None else median - 5,
    "q90": hi if hi is not None else median + 5,
  }


def test_score_of_a_perfect_forecast():
  points = [_pt(100.0, 100.0, 99.0), _pt(102.0, 102.0, 100.0)]
  out = score(points)
  assert out["mae"] == pytest.approx(0.0)
  assert out["rmse"] == pytest.approx(0.0)
  assert out["direction_acc"] == pytest.approx(100.0)
  assert out["skill"] == pytest.approx(0.0)


def test_skill_compares_against_the_price_does_not_move_baseline():
  # Model is off by 1; persistence is off by 2 -> skill 0.5.
  points = [_pt(101.0, 102.0, 100.0)]
  out = score(points)
  assert out["persistence_mae"] == pytest.approx(2.0)
  assert out["skill"] == pytest.approx(0.5)


def test_direction_accuracy_ignores_unmoved_bars():
  points = [
    _pt(101.0, 100.0, 100.0),  # market did not move: carries no direction
    _pt(101.0, 102.0, 100.0),  # called up, went up
  ]
  assert score(points)["direction_acc"] == pytest.approx(100.0)


def test_direction_accuracy_counts_a_wrong_call():
  points = [_pt(101.0, 99.0, 100.0), _pt(101.0, 102.0, 100.0)]
  assert score(points)["direction_acc"] == pytest.approx(50.0)


def test_interval_coverage_counts_only_points_inside_the_band():
  points = [_pt(100.0, 101.0, 100.0), _pt(100.0, 200.0, 100.0)]
  assert score(points)["coverage_80"] == pytest.approx(50.0)


def test_bias_is_signed():
  assert score([_pt(105.0, 100.0, 100.0)])["bias"] == pytest.approx(5.0)


def test_empty_score_is_all_none_rather_than_zero():
  out = score([])
  assert out["n"] == 0
  assert all(out[k] is None for k in ("mae", "rmse", "mape", "skill", "direction_acc"))
