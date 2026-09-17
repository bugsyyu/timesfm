"""The 13 covariate channels, computed on a shared 5-minute grid.

Two rules govern everything in this module:

1. **No look-ahead.** Every channel value at grid timestamp `T` is derived only
   from data that was final by the end of the bucket starting at `T`. Both the
   candles and the rubik statistics are stamped with their *bucket open* time
   and describe the interval `[T, T + 5min)`, so a straight join on timestamp is
   causally sound -- provided the still-open bucket is dropped, which
   `drop_incomplete` does.

2. **No partial warmup.** Indicators are computed over a longer history than the
   context window and then sliced down, so the RSI at the left edge of the
   window is a real 14-bar RSI rather than a value seeded mid-computation.

All functions take and return plain numpy arrays so they can be unit-tested
without touching the network.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import config as cfg
from .okx import Candle, Sample

# --- Primitives -------------------------------------------------------------


def ema(values: np.ndarray, span: int) -> np.ndarray:
  """Exponential moving average with alpha = 2/(span+1), seeded at values[0]."""
  values = np.asarray(values, dtype=np.float64)
  if values.size == 0:
    return values.copy()
  alpha = 2.0 / (span + 1.0)
  out = np.empty_like(values)
  out[0] = values[0]
  for i in range(1, values.size):
    out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
  return out


def wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
  """Wilder's smoothing (alpha = 1/period), seeded with the first SMA.

  Entries before the seed are NaN, which keeps the warmup region explicit
  instead of silently blending in a half-formed average.
  """
  values = np.asarray(values, dtype=np.float64)
  out = np.full(values.shape, np.nan)
  if values.size < period:
    return out
  seed = float(np.mean(values[:period]))
  out[period - 1] = seed
  prev = seed
  for i in range(period, values.size):
    prev = (prev * (period - 1) + values[i]) / period
    out[i] = prev
  return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
  values = np.asarray(values, dtype=np.float64)
  out = np.full(values.shape, np.nan)
  if values.size < window:
    return out
  csum = np.cumsum(np.insert(values, 0, 0.0))
  out[window - 1 :] = (csum[window:] - csum[:-window]) / window
  return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
  """Rolling sample standard deviation (ddof=1)."""
  values = np.asarray(values, dtype=np.float64)
  out = np.full(values.shape, np.nan)
  if values.size < window or window < 2:
    return out
  for i in range(window - 1, values.size):
    out[i] = float(np.std(values[i - window + 1 : i + 1], ddof=1))
  return out


# --- The 8 technical channels ----------------------------------------------


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
  """Wilder's RSI in [0, 100]."""
  close = np.asarray(close, dtype=np.float64)
  out = np.full(close.shape, np.nan)
  if close.size < period + 1:
    return out
  delta = np.diff(close)
  gains = wilder_smooth(np.maximum(delta, 0.0), period)
  losses = wilder_smooth(np.maximum(-delta, 0.0), period)
  # `delta[i]` describes the move into close[i+1], so shift by one.
  with np.errstate(divide="ignore", invalid="ignore"):
    rs = np.where(losses > 0, gains / losses, np.inf)
  values = np.where(np.isfinite(rs), 100.0 - 100.0 / (1.0 + rs), 100.0)
  values = np.where(np.isnan(gains) | np.isnan(losses), np.nan, values)
  out[1:] = values
  return out


def macd_hist(
  close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> np.ndarray:
  """MACD histogram (DIF - DEA), expressed as a percentage of price.

  Dividing by close keeps the channel comparable across price levels, which
  matters here because ZEC can move tens of percent inside a single session.
  """
  close = np.asarray(close, dtype=np.float64)
  dif = ema(close, fast) - ema(close, slow)
  dea = ema(dif, signal)
  hist = dif - dea
  return np.where(close > 0, hist / close * 100.0, np.nan)


def ema_bias(close: np.ndarray, span: int = 20) -> np.ndarray:
  """Percentage deviation of price from its EMA (乖离率)."""
  close = np.asarray(close, dtype=np.float64)
  baseline = ema(close, span)
  return np.where(baseline > 0, (close - baseline) / baseline * 100.0, np.nan)


def atr_pct(
  high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
  """Wilder's ATR as a percentage of close."""
  high = np.asarray(high, dtype=np.float64)
  low = np.asarray(low, dtype=np.float64)
  close = np.asarray(close, dtype=np.float64)
  out = np.full(close.shape, np.nan)
  if close.size < 2:
    return out
  prev_close = close[:-1]
  true_range = np.maximum.reduce(
    [
      high[1:] - low[1:],
      np.abs(high[1:] - prev_close),
      np.abs(low[1:] - prev_close),
    ]
  )
  atr = wilder_smooth(true_range, period)
  out[1:] = np.where(close[1:] > 0, atr / close[1:] * 100.0, np.nan)
  return out


def momentum_pct(close: np.ndarray, lag_bars: int) -> np.ndarray:
  """Percent change over `lag_bars` bars."""
  close = np.asarray(close, dtype=np.float64)
  out = np.full(close.shape, np.nan)
  if close.size <= lag_bars:
    return out
  past = close[:-lag_bars]
  out[lag_bars:] = np.where(past > 0, (close[lag_bars:] / past - 1.0) * 100.0, np.nan)
  return out


def realized_vol(close: np.ndarray, window: int = 12) -> np.ndarray:
  """Standard deviation of 5m log returns, scaled to an hourly figure in percent."""
  close = np.asarray(close, dtype=np.float64)
  out = np.full(close.shape, np.nan)
  if close.size < 2:
    return out
  with np.errstate(divide="ignore", invalid="ignore"):
    log_ret = np.diff(np.log(np.where(close > 0, close, np.nan)))
  sigma = rolling_std(log_ret, window)
  bars_per_hour = 3_600_000 / cfg.BAR_MS
  out[1:] = sigma * math.sqrt(bars_per_hour) * 100.0
  return out


def volume_ratio(volume: np.ndarray, window: int = 12) -> np.ndarray:
  """Current bar volume over the mean of the *preceding* `window` bars (量比).

  The current bar is excluded from its own baseline, so 1.0 means "trading at
  the pace of the last hour" rather than a value structurally pulled toward 1.
  """
  volume = np.asarray(volume, dtype=np.float64)
  out = np.full(volume.shape, np.nan)
  if volume.size <= window:
    return out
  baseline = rolling_mean(volume, window)[:-1]  # mean of bars ending one bar back
  current = volume[1:]
  with np.errstate(divide="ignore", invalid="ignore"):
    out[1:] = np.where(baseline > 0, current / baseline, np.nan)
  return out


# --- Alignment --------------------------------------------------------------


def drop_incomplete(samples: list[Sample], now_ms: int, bar_ms: int = cfg.BAR_MS) -> list[Sample]:
  """Removes buckets whose interval has not finished yet.

  The rubik endpoints always include the bucket currently being filled. Joining
  it to the grid would feed the model a partial 5-minute aggregate, which is
  both noisy and a mild look-ahead (it reflects trades after the bar we treat as
  the forecast origin).
  """
  return [s for s in samples if s.ts + bar_ms <= now_ms]


def asof(grid_ts: np.ndarray, samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
  """As-of joins `samples` onto `grid_ts`, carrying the last value forward.

  Returns (values, is_observed) where `is_observed` marks grid points that land
  exactly on a sample timestamp -- the dashboard reports the share of each
  channel that is genuinely observed rather than carried forward.
  """
  values = np.full(grid_ts.shape, np.nan)
  observed = np.zeros(grid_ts.shape, dtype=bool)
  if not samples:
    return values, observed

  sample_ts = np.array([s.ts for s in samples], dtype=np.int64)
  sample_val = np.array([s.value for s in samples], dtype=np.float64)
  order = np.argsort(sample_ts, kind="stable")
  sample_ts, sample_val = sample_ts[order], sample_val[order]

  idx = np.searchsorted(sample_ts, grid_ts, side="right") - 1
  valid = idx >= 0
  values[valid] = sample_val[idx[valid]]
  observed[valid] = sample_ts[idx[valid]] == grid_ts[valid]
  return values, observed


def fill_gaps(values: np.ndarray) -> np.ndarray:
  """Forward-fills, then back-fills, any remaining NaNs."""
  values = np.asarray(values, dtype=np.float64).copy()
  n = values.size
  if n == 0 or np.all(np.isnan(values)):
    return np.zeros(n)
  # forward
  last = np.nan
  for i in range(n):
    if np.isnan(values[i]):
      values[i] = last
    else:
      last = values[i]
  # backward, for the head before the first observation
  nxt = np.nan
  for i in range(n - 1, -1, -1):
    if np.isnan(values[i]):
      values[i] = nxt
    else:
      nxt = values[i]
  return values


# --- Assembly ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FeatureWindow:
  """A fully aligned 5.9 hour window, ready to hand to the model."""

  grid_ts: np.ndarray  # (CONTEXT_BARS,) int64, bucket open times
  close: np.ndarray  # (CONTEXT_BARS,) float32, the forecast target
  high: np.ndarray
  low: np.ndarray
  open_: np.ndarray
  volume_quote: np.ndarray
  channels: dict[str, np.ndarray]  # key -> (CONTEXT_BARS,) float32
  coverage: dict[str, float]  # key -> share of grid points actually observed

  @property
  def origin_ts(self) -> int:
    """Timestamp of the last closed bar: the forecast origin."""
    return int(self.grid_ts[-1])

  @property
  def last_close(self) -> float:
    return float(self.close[-1])

  def covariate_matrix(self, keys: tuple[str, ...]) -> np.ndarray | None:
    """Stacks the requested channels into the (num_channels, context) layout."""
    if not keys:
      return None
    return np.stack([self.channels[k] for k in keys]).astype(np.float32)


def build_feature_window(
  candles: list[Candle],
  *,
  now_ms: int,
  taker_buy_share: list[Sample],
  long_short_account_ratio: list[Sample],
  margin_loan_ratio: list[Sample],
  open_interest: list[Sample],
  funding_bp: list[Sample],
  context_bars: int = cfg.CONTEXT_BARS,
) -> FeatureWindow:
  """Builds the aligned covariate window from one snapshot of raw series.

  `candles` must be closed bars, oldest first, and should extend `WARMUP_BARS`
  further back than `context_bars` so the indicators are fully warmed.
  """
  if len(candles) < context_bars:
    raise ValueError(
      f"need at least {context_bars} closed candles, got {len(candles)}"
    )

  ts = np.array([c.ts for c in candles], dtype=np.int64)
  open_ = np.array([c.open for c in candles], dtype=np.float64)
  high = np.array([c.high for c in candles], dtype=np.float64)
  low = np.array([c.low for c in candles], dtype=np.float64)
  close = np.array([c.close for c in candles], dtype=np.float64)
  volume = np.array([c.volume_quote for c in candles], dtype=np.float64)

  # Technical indicators over the full fetched history, sliced afterwards.
  wide = {
    "rsi_14": rsi(close, 14),
    "macd_hist": macd_hist(close),
    "ema_bias": ema_bias(close, 20),
    "atr_pct": atr_pct(high, low, close, 14),
    "mom_15m": momentum_pct(close, 3),  # 3 bars * 5min = 15min
    "mom_60m": momentum_pct(close, 12),  # 12 bars * 5min = 60min
    "realized_vol": realized_vol(close, 12),
    "volume_ratio": volume_ratio(volume, 12),
  }

  sl = slice(len(candles) - context_bars, len(candles))
  grid_ts = ts[sl]

  channels: dict[str, np.ndarray] = {}
  coverage: dict[str, float] = {}
  for key, series in wide.items():
    window = series[sl]
    coverage[key] = float(np.mean(~np.isnan(window)))
    channels[key] = fill_gaps(window).astype(np.float32)

  # Derivative channels: drop the in-progress bucket, then as-of join.
  derivative_sources: dict[str, list[Sample]] = {
    "funding_rate": funding_bp,
    "taker_buy_share": drop_incomplete(taker_buy_share, now_ms),
    "long_short_account_ratio": drop_incomplete(long_short_account_ratio, now_ms),
    "margin_loan_ratio": drop_incomplete(margin_loan_ratio, now_ms),
    "open_interest": drop_incomplete(open_interest, now_ms),
  }
  for key, samples in derivative_sources.items():
    values, observed = asof(grid_ts, samples)
    if key == "funding_rate":
      # Funding settles every 8h, so a carried-forward value is the correct
      # value, not a gap. Coverage here means "we had any rate at all".
      coverage[key] = float(np.mean(~np.isnan(values)))
    else:
      coverage[key] = float(np.mean(observed))
    channels[key] = fill_gaps(values).astype(np.float32)

  return FeatureWindow(
    grid_ts=grid_ts,
    close=close[sl].astype(np.float32),
    high=high[sl].astype(np.float32),
    low=low[sl].astype(np.float32),
    open_=open_[sl].astype(np.float32),
    volume_quote=volume[sl].astype(np.float32),
    channels=channels,
    coverage=coverage,
  )
