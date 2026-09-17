"""TimesFM 3.0 inference for the three side-by-side configurations.

One subtlety drives the shape of this module: the three configurations cannot
share a single `predict_batch` call. TimesFM stacks a batch's covariates into
one array, so configurations with different channel counts (0, 8 and 13) cannot
be batched together -- and a query that passes `None` inside a batch where
others pass covariates is given a *zero-filled* covariate block rather than no
covariates at all, which is not the same model input. Each configuration
therefore gets its own call, and config A really does run covariate-free.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
import time

import numpy as np

from . import config as cfg
from .features import FeatureWindow

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ConfigForecast:
  """A single configuration's forecast for one origin."""

  config_id: str
  origin_ts: int
  horizon_bars: int
  target_ts: np.ndarray  # (horizon,) int64
  median: np.ndarray  # (horizon,) float64
  quantiles: np.ndarray  # (horizon, 9) float64
  latency_ms: float


class Engine(abc.ABC):
  """Common interface so the service does not care which engine is loaded."""

  #: Short identifier stored with every forecast row and shown in the UI.
  name: str = "engine"
  #: True only for the real foundation model; the dashboard warns when False.
  is_foundation_model: bool = False

  @abc.abstractmethod
  def forecast(
    self, window: FeatureWindow, config: cfg.ForecastConfig, horizon_bars: int
  ) -> ConfigForecast:
    ...

  def forecast_all(
    self,
    window: FeatureWindow,
    horizon_bars: int = cfg.HORIZON_BARS,
    configs: tuple[cfg.ForecastConfig, ...] = cfg.CONFIGS,
  ) -> list[ConfigForecast]:
    return [self.forecast(window, c, horizon_bars) for c in configs]

  @staticmethod
  def target_timestamps(origin_ts: int, horizon_bars: int) -> np.ndarray:
    """Bucket open times of the bars being predicted, first one after origin."""
    return np.array(
      [origin_ts + (i + 1) * cfg.BAR_MS for i in range(horizon_bars)], dtype=np.int64
    )


class TimesFm3Engine(Engine):
  """Runs the real `google/timesfm-3.0-pytorch` checkpoint."""

  name = "timesfm-3.0"
  is_foundation_model = True

  def __init__(
    self,
    checkpoint_path: str = "google/timesfm-3.0-pytorch",
    device: str | None = None,
    per_core_batch_size: int = 1,
  ) -> None:
    from timesfm3 import ModelConfig, TimesFM3Forecaster  # imported lazily: needs torch

    started = time.perf_counter()
    self._forecaster = TimesFM3Forecaster(
      ModelConfig(
        checkpoint_path=checkpoint_path,
        per_core_batch_size=per_core_batch_size,
        device=device,
      )
    )
    self.checkpoint_path = checkpoint_path
    self.device = str(self._forecaster.device)
    _LOG.info(
      "loaded %s on %s in %.1fs",
      checkpoint_path,
      self.device,
      time.perf_counter() - started,
    )

  def forecast(
    self, window: FeatureWindow, config: cfg.ForecastConfig, horizon_bars: int
  ) -> ConfigForecast:
    covariates = window.covariate_matrix(config.channels)
    target = window.close.astype(np.float32)

    started = time.perf_counter()
    output = self._forecaster.predict(
      context=target,
      horizon=horizon_bars,
      past_only_covariates=covariates,
      return_quantiles=True,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    median = np.asarray(output.forecast, dtype=np.float64).reshape(-1)[:horizon_bars]
    quantiles = np.asarray(output.quantiles, dtype=np.float64)
    quantiles = quantiles.reshape(-1, len(cfg.QUANTILE_LEVELS))[:horizon_bars]

    return ConfigForecast(
      config_id=config.id,
      origin_ts=window.origin_ts,
      horizon_bars=horizon_bars,
      target_ts=self.target_timestamps(window.origin_ts, horizon_bars),
      median=median,
      quantiles=quantiles,
      latency_ms=latency_ms,
    )


class RandomWalkEngine(Engine):
  """Development stand-in for when the 3.0 weights are not available.

  It emits a flat random-walk forecast (last close carried forward, with bands
  widening as sqrt(h) at the window's realized volatility). It exists so the
  dashboard can be developed without the checkpoint; it is *not* a forecast, and
  every surface that shows it says so. Select it explicitly with
  `--engine random-walk` -- it is never a silent fallback.
  """

  name = "random-walk-baseline"
  is_foundation_model = False

  def forecast(
    self, window: FeatureWindow, config: cfg.ForecastConfig, horizon_bars: int
  ) -> ConfigForecast:
    started = time.perf_counter()
    last = float(window.close[-1])
    log_ret = np.diff(np.log(window.close.astype(np.float64)))
    sigma = float(np.std(log_ret, ddof=1)) if log_ret.size > 1 else 0.002

    steps = np.arange(1, horizon_bars + 1, dtype=np.float64)
    median = np.full(horizon_bars, last)
    # Normal quantiles for the nine deciles TimesFM reports.
    z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])
    spread = sigma * np.sqrt(steps)[:, None] * z[None, :]
    quantiles = last * np.exp(spread)

    return ConfigForecast(
      config_id=config.id,
      origin_ts=window.origin_ts,
      horizon_bars=horizon_bars,
      target_ts=self.target_timestamps(window.origin_ts, horizon_bars),
      median=median,
      quantiles=quantiles,
      latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def build_engine(
  kind: str = "timesfm",
  *,
  checkpoint_path: str = "google/timesfm-3.0-pytorch",
  device: str | None = None,
) -> Engine:
  """Creates the requested engine, failing loudly rather than downgrading."""
  if kind == "timesfm":
    return TimesFm3Engine(checkpoint_path=checkpoint_path, device=device)
  if kind == "random-walk":
    _LOG.warning(
      "starting with the random-walk baseline: output is NOT a TimesFM forecast"
    )
    return RandomWalkEngine()
  raise ValueError(f"unknown engine {kind!r}; expected 'timesfm' or 'random-walk'")
