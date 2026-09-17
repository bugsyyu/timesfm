"""Unit tests for the indicator maths and the grid alignment.

These run without network access or model weights: everything here is pure
numpy over synthetic series.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from zecfm import config as cfg
from zecfm import features as ft
from zecfm.okx import Candle, Sample


# --- primitives -------------------------------------------------------------


def test_ema_matches_manual_recursion():
  values = np.array([1.0, 2.0, 3.0, 4.0])
  out = ft.ema(values, span=3)  # alpha = 0.5
  expected = [1.0, 1.5, 2.25, 3.125]
  assert out == pytest.approx(expected)


def test_wilder_smooth_seeds_with_sma_and_leaves_warmup_nan():
  values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  out = ft.wilder_smooth(values, period=3)
  assert np.isnan(out[:2]).all()
  assert out[2] == pytest.approx(2.0)  # SMA of 1,2,3
  assert out[3] == pytest.approx((2.0 * 2 + 4.0) / 3)


def test_rolling_mean_and_std_align_to_window_end():
  values = np.arange(1.0, 6.0)
  assert ft.rolling_mean(values, 3)[2] == pytest.approx(2.0)
  assert np.isnan(ft.rolling_mean(values, 3)[1])
  assert ft.rolling_std(values, 3)[2] == pytest.approx(1.0)


# --- technical channels -----------------------------------------------------


def test_rsi_saturates_on_a_monotone_rally():
  close = np.linspace(100.0, 130.0, 40)
  out = ft.rsi(close, 14)
  assert out[-1] == pytest.approx(100.0)
  assert np.isnan(out[0])


def test_rsi_bottoms_out_on_a_monotone_selloff():
  close = np.linspace(130.0, 100.0, 40)
  assert ft.rsi(close, 14)[-1] == pytest.approx(0.0)


def test_rsi_sits_mid_range_on_alternating_moves():
  close = 100 + np.tile([0.0, 1.0], 40)
  out = ft.rsi(close, 14)
  assert 30.0 < out[-1] < 70.0


def test_macd_hist_is_scale_free():
  """Doubling the price level must not double the channel."""
  base = 100 + np.sin(np.linspace(0, 8, 120)) * 3
  a = ft.macd_hist(base)
  b = ft.macd_hist(base * 10.0)
  assert a[-1] == pytest.approx(b[-1], rel=1e-9)


def test_ema_bias_is_zero_on_a_flat_series():
  close = np.full(60, 42.0)
  assert ft.ema_bias(close, 20)[-1] == pytest.approx(0.0)


def test_atr_pct_on_constant_range_bars():
  # Every bar spans exactly 2.0 with no gaps, so ATR converges to 2.0.
  n = 60
  close = np.full(n, 100.0)
  high = close + 1.0
  low = close - 1.0
  out = ft.atr_pct(high, low, close, 14)
  assert out[-1] == pytest.approx(2.0, rel=1e-6)  # 2.0 / 100 * 100


def test_momentum_uses_the_right_lag():
  close = np.array([100.0, 101.0, 102.0, 103.0, 110.0])
  out = ft.momentum_pct(close, 3)
  # 110 vs close[1] = 101
  assert out[-1] == pytest.approx((110.0 / 101.0 - 1) * 100)
  assert np.isnan(out[:3]).all()


def test_realized_vol_is_zero_on_a_flat_series_and_positive_otherwise():
  flat = np.full(40, 50.0)
  assert ft.realized_vol(flat, 12)[-1] == pytest.approx(0.0)
  noisy = 50.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 40)))
  assert ft.realized_vol(noisy, 12)[-1] > 0.0


def test_realized_vol_scales_to_an_hourly_figure():
  rng = np.random.default_rng(7)
  steps = rng.normal(0, 0.01, 400)
  close = 100.0 * np.exp(np.cumsum(steps))
  out = ft.realized_vol(close, 12)[-1]
  bars_per_hour = 3_600_000 / cfg.BAR_MS
  assert out == pytest.approx(0.01 * math.sqrt(bars_per_hour) * 100, rel=0.45)


def test_volume_ratio_excludes_the_current_bar_from_its_own_baseline():
  volume = np.concatenate([np.full(12, 100.0), [300.0]])
  out = ft.volume_ratio(volume, 12)
  assert out[-1] == pytest.approx(3.0)


def test_volume_ratio_is_one_when_volume_is_steady():
  volume = np.full(30, 77.0)
  assert ft.volume_ratio(volume, 12)[-1] == pytest.approx(1.0)


# --- alignment --------------------------------------------------------------


def test_drop_incomplete_removes_the_in_progress_bucket():
  now = 1_000_000 + cfg.BAR_MS * 3 + 1000
  samples = [Sample(1_000_000 + cfg.BAR_MS * i, float(i)) for i in range(5)]
  kept = ft.drop_incomplete(samples, now)
  # Buckets 0,1,2 have closed; bucket 3 is still filling and 4 is in the future.
  assert [s.ts for s in kept] == [1_000_000 + cfg.BAR_MS * i for i in range(3)]


def test_asof_carries_forward_and_flags_observed_points():
  grid = np.array([100, 200, 300, 400], dtype=np.int64)
  samples = [Sample(100, 1.0), Sample(300, 3.0)]
  values, observed = ft.asof(grid, samples)
  assert values.tolist() == [1.0, 1.0, 3.0, 3.0]
  assert observed.tolist() == [True, False, True, False]


def test_asof_leaves_nan_before_the_first_sample():
  grid = np.array([100, 200], dtype=np.int64)
  values, observed = ft.asof(grid, [Sample(150, 5.0)])
  assert np.isnan(values[0])
  assert values[1] == 5.0
  assert observed.tolist() == [False, False]


def test_fill_gaps_forward_then_backward():
  values = np.array([np.nan, 1.0, np.nan, 2.0, np.nan])
  assert ft.fill_gaps(values).tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]


def test_fill_gaps_on_an_all_nan_series_returns_zeros():
  assert ft.fill_gaps(np.array([np.nan, np.nan])).tolist() == [0.0, 0.0]


# --- window assembly --------------------------------------------------------


def _synthetic_candles(n: int, start_ts: int = 1_700_000_000_000) -> list[Candle]:
  rng = np.random.default_rng(42)
  price = 100.0
  out = []
  for i in range(n):
    price *= math.exp(rng.normal(0, 0.003))
    out.append(
      Candle(
        ts=start_ts + i * cfg.BAR_MS,
        open=price * 0.999,
        high=price * 1.004,
        low=price * 0.996,
        close=price,
        volume_quote=abs(rng.normal(1_000_000, 200_000)),
      )
    )
  return out


def _derivative_samples(candles: list[Candle], value: float) -> list[Sample]:
  return [Sample(ts=c.ts, value=value + i * 0.01) for i, c in enumerate(candles)]


def _build(candles: list[Candle], **overrides):
  now = candles[-1].ts + cfg.BAR_MS  # the last candle has just closed
  kwargs = dict(
    now_ms=now,
    taker_buy_share=_derivative_samples(candles, 50.0),
    long_short_account_ratio=_derivative_samples(candles, 0.9),
    margin_loan_ratio=_derivative_samples(candles, 1.1),
    open_interest=_derivative_samples(candles, 200.0),
    funding_bp=[Sample(candles[0].ts, -2.0)],
  )
  kwargs.update(overrides)
  return ft.build_feature_window(candles, **kwargs)


def test_window_has_all_thirteen_channels_at_the_right_length():
  window = _build(_synthetic_candles(cfg.CONTEXT_BARS + cfg.WARMUP_BARS))
  assert set(window.channels) == set(cfg.ALL_KEYS)
  assert len(window.channels) == 13
  for key in cfg.ALL_KEYS:
    assert window.channels[key].shape == (cfg.CONTEXT_BARS,)
    assert np.isfinite(window.channels[key]).all(), key
  assert window.grid_ts.shape == (cfg.CONTEXT_BARS,)
  assert window.close.shape == (cfg.CONTEXT_BARS,)


def test_window_covers_the_documented_five_point_nine_hours():
  window = _build(_synthetic_candles(cfg.CONTEXT_BARS + cfg.WARMUP_BARS))
  covered_ms = cfg.CONTEXT_BARS * cfg.BAR_MS
  assert covered_ms / 3_600_000 == pytest.approx(5.9167, abs=0.01)
  # The grid is contiguous on the 5m boundary.
  assert np.all(np.diff(window.grid_ts) == cfg.BAR_MS)


def test_covariate_matrix_shapes_match_each_configuration():
  window = _build(_synthetic_candles(cfg.CONTEXT_BARS + cfg.WARMUP_BARS))
  assert window.covariate_matrix(()) is None
  assert window.covariate_matrix(cfg.TECHNICAL_KEYS).shape == (8, cfg.CONTEXT_BARS)
  assert window.covariate_matrix(cfg.ALL_KEYS).shape == (13, cfg.CONTEXT_BARS)


def test_channels_are_not_affected_by_later_bars():
  """The no-look-ahead guarantee: appending a bar must not rewrite history."""
  candles = _synthetic_candles(cfg.CONTEXT_BARS + cfg.WARMUP_BARS + 1)
  earlier = _build(candles[:-1])
  later = _build(candles)

  # The two windows overlap on every grid point of `earlier` except its first,
  # since the window slides by one bar.
  shared = np.intersect1d(earlier.grid_ts, later.grid_ts)
  assert shared.size == cfg.CONTEXT_BARS - 1
  ea = np.isin(earlier.grid_ts, shared)
  la = np.isin(later.grid_ts, shared)
  for key in cfg.ALL_KEYS:
    np.testing.assert_allclose(
      earlier.channels[key][ea], later.channels[key][la], rtol=1e-5, atol=1e-5,
      err_msg=f"{key} changed retroactively",
    )


def test_incomplete_derivative_bucket_never_reaches_the_grid():
  candles = _synthetic_candles(cfg.CONTEXT_BARS + cfg.WARMUP_BARS)
  now = candles[-1].ts + cfg.BAR_MS
  poisoned = _derivative_samples(candles, 50.0) + [
    Sample(ts=candles[-1].ts + cfg.BAR_MS, value=999.0)  # the bucket still filling
  ]
  window = _build(candles, taker_buy_share=poisoned)
  assert float(window.channels["taker_buy_share"].max()) < 999.0


def test_coverage_reports_carried_forward_points():
  candles = _synthetic_candles(cfg.CONTEXT_BARS + cfg.WARMUP_BARS)
  sparse = [s for i, s in enumerate(_derivative_samples(candles, 50.0)) if i % 2 == 0]
  window = _build(candles, open_interest=sparse)
  assert 0.4 < window.coverage["open_interest"] < 0.75
  assert window.coverage["taker_buy_share"] == pytest.approx(1.0)


def test_origin_is_the_last_closed_bar():
  candles = _synthetic_candles(cfg.CONTEXT_BARS + cfg.WARMUP_BARS)
  window = _build(candles)
  assert window.origin_ts == candles[-1].ts
  assert window.last_close == pytest.approx(candles[-1].close, rel=1e-6)


def test_too_few_candles_is_an_error_not_a_short_window():
  with pytest.raises(ValueError, match="at least"):
    _build(_synthetic_candles(cfg.CONTEXT_BARS - 1))
