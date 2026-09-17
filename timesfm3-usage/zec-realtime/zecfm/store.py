"""SQLite persistence and scoring.

Every forecast is written the moment it is produced, before the bars it predicts
exist. Scoring then joins forecast points to bars that arrived *later*, so a
prediction can never be graded against data it could have seen. That ordering is
the whole point of the comparison: the accuracy table is out-of-sample by
construction, not by convention.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Iterable

import numpy as np

from . import config as cfg
from .features import FeatureWindow
from .forecaster import ConfigForecast
from .okx import Candle, Sample

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
  ts            INTEGER PRIMARY KEY,
  open          REAL NOT NULL,
  high          REAL NOT NULL,
  low           REAL NOT NULL,
  close         REAL NOT NULL,
  volume_quote  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS forecasts (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  origin_ts     INTEGER NOT NULL,
  config_id     TEXT    NOT NULL,
  created_ms    INTEGER NOT NULL,
  engine        TEXT    NOT NULL,
  horizon_bars  INTEGER NOT NULL,
  context_bars  INTEGER NOT NULL,
  origin_close  REAL    NOT NULL,
  latency_ms    REAL,
  UNIQUE (origin_ts, config_id)
);

CREATE TABLE IF NOT EXISTS forecast_points (
  forecast_id   INTEGER NOT NULL REFERENCES forecasts(id) ON DELETE CASCADE,
  step          INTEGER NOT NULL,
  target_ts     INTEGER NOT NULL,
  median        REAL    NOT NULL,
  q10           REAL    NOT NULL,
  q90           REAL    NOT NULL,
  quantiles     TEXT    NOT NULL,
  PRIMARY KEY (forecast_id, step)
);
CREATE INDEX IF NOT EXISTS idx_points_target ON forecast_points(target_ts);

CREATE TABLE IF NOT EXISTS channel_windows (
  origin_ts     INTEGER PRIMARY KEY,
  payload       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS funding_samples (
  ts            INTEGER PRIMARY KEY,
  rate_bp       REAL NOT NULL
);
"""


class Store:
  """Thread-safe SQLite wrapper. One connection guarded by a lock is plenty for
  a single-instrument service polling every 20 seconds."""

  def __init__(self, path: str) -> None:
    self.path = path
    self._lock = threading.Lock()
    self._conn = sqlite3.connect(path, check_same_thread=False)
    self._conn.row_factory = sqlite3.Row
    with self._lock:
      self._conn.execute("PRAGMA journal_mode=WAL")
      self._conn.execute("PRAGMA synchronous=NORMAL")
      self._conn.executescript(_SCHEMA)
      self._conn.commit()

  def close(self) -> None:
    with self._lock:
      self._conn.close()

  # --- Writes ---------------------------------------------------------------

  def upsert_bars(self, candles: Iterable[Candle]) -> int:
    rows = [
      (c.ts, c.open, c.high, c.low, c.close, c.volume_quote) for c in candles
    ]
    if not rows:
      return 0
    with self._lock:
      self._conn.executemany(
        "INSERT INTO bars (ts, open, high, low, close, volume_quote)"
        " VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(ts) DO UPDATE SET"
        " open=excluded.open, high=excluded.high, low=excluded.low,"
        " close=excluded.close, volume_quote=excluded.volume_quote",
        rows,
      )
      self._conn.commit()
    return len(rows)

  def record_funding(self, ts: int, rate_bp: float) -> None:
    """Records a live funding observation, bucketed to the 5m grid.

    Funding updates continuously between settlements; bucketing the samples to
    bar boundaries is what lets the channel join the grid like any other series.
    """
    bucket = ts - (ts % cfg.BAR_MS)
    with self._lock:
      self._conn.execute(
        "INSERT INTO funding_samples (ts, rate_bp) VALUES (?, ?)"
        " ON CONFLICT(ts) DO UPDATE SET rate_bp=excluded.rate_bp",
        (bucket, rate_bp),
      )
      self._conn.commit()

  def funding_samples(self, since_ms: int) -> list[Sample]:
    with self._lock:
      rows = self._conn.execute(
        "SELECT ts, rate_bp FROM funding_samples WHERE ts >= ? ORDER BY ts",
        (since_ms,),
      ).fetchall()
    return [Sample(ts=r["ts"], value=r["rate_bp"]) for r in rows]

  def save_forecast(
    self,
    forecast: ConfigForecast,
    *,
    engine: str,
    origin_close: float,
    context_bars: int,
    created_ms: int,
  ) -> int | None:
    """Persists one configuration's forecast. Returns None if it already exists."""
    with self._lock:
      cursor = self._conn.execute(
        "INSERT OR IGNORE INTO forecasts"
        " (origin_ts, config_id, created_ms, engine, horizon_bars, context_bars,"
        "  origin_close, latency_ms)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
          forecast.origin_ts,
          forecast.config_id,
          created_ms,
          engine,
          forecast.horizon_bars,
          context_bars,
          origin_close,
          forecast.latency_ms,
        ),
      )
      if cursor.rowcount == 0:
        self._conn.commit()
        return None
      forecast_id = int(cursor.lastrowid)
      self._conn.executemany(
        "INSERT INTO forecast_points"
        " (forecast_id, step, target_ts, median, q10, q90, quantiles)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
          (
            forecast_id,
            step + 1,
            int(forecast.target_ts[step]),
            float(forecast.median[step]),
            float(forecast.quantiles[step, 0]),
            float(forecast.quantiles[step, 8]),
            json.dumps([round(float(v), 6) for v in forecast.quantiles[step]]),
          )
          for step in range(forecast.horizon_bars)
        ],
      )
      self._conn.commit()
    return forecast_id

  def save_channel_window(self, window: FeatureWindow) -> None:
    payload = {
      "grid_ts": [int(t) for t in window.grid_ts],
      "close": [round(float(v), 6) for v in window.close],
      "channels": {
        k: [round(float(v), 6) for v in series]
        for k, series in window.channels.items()
      },
      "coverage": {k: round(float(v), 4) for k, v in window.coverage.items()},
    }
    with self._lock:
      self._conn.execute(
        "INSERT INTO channel_windows (origin_ts, payload) VALUES (?, ?)"
        " ON CONFLICT(origin_ts) DO UPDATE SET payload=excluded.payload",
        (window.origin_ts, json.dumps(payload)),
      )
      self._conn.commit()

  # --- Reads ----------------------------------------------------------------

  def has_forecast(self, origin_ts: int, config_id: str) -> bool:
    with self._lock:
      row = self._conn.execute(
        "SELECT 1 FROM forecasts WHERE origin_ts = ? AND config_id = ?",
        (origin_ts, config_id),
      ).fetchone()
    return row is not None

  def latest_origin(self) -> int | None:
    with self._lock:
      row = self._conn.execute("SELECT MAX(origin_ts) AS m FROM forecasts").fetchone()
    return None if row is None or row["m"] is None else int(row["m"])

  def recent_bars(self, limit: int = 120) -> list[dict[str, Any]]:
    with self._lock:
      rows = self._conn.execute(
        "SELECT ts, open, high, low, close, volume_quote FROM bars"
        " ORDER BY ts DESC LIMIT ?",
        (limit,),
      ).fetchall()
    return [dict(r) for r in reversed(rows)]

  def latest_forecast(self, config_id: str) -> dict[str, Any] | None:
    """Newest forecast for a configuration, with its points."""
    with self._lock:
      head = self._conn.execute(
        "SELECT * FROM forecasts WHERE config_id = ? ORDER BY origin_ts DESC LIMIT 1",
        (config_id,),
      ).fetchone()
      if head is None:
        return None
      points = self._conn.execute(
        "SELECT step, target_ts, median, q10, q90 FROM forecast_points"
        " WHERE forecast_id = ? ORDER BY step",
        (head["id"],),
      ).fetchall()
    return {**dict(head), "points": [dict(p) for p in points]}

  def channel_window(self, origin_ts: int) -> dict[str, Any] | None:
    with self._lock:
      row = self._conn.execute(
        "SELECT payload FROM channel_windows WHERE origin_ts = ?", (origin_ts,)
      ).fetchone()
    return None if row is None else json.loads(row["payload"])

  def latest_channel_window(self) -> dict[str, Any] | None:
    with self._lock:
      row = self._conn.execute(
        "SELECT payload FROM channel_windows ORDER BY origin_ts DESC LIMIT 1"
      ).fetchone()
    return None if row is None else json.loads(row["payload"])

  def resolved_points(
    self, config_id: str, *, limit: int = 5000, step: int | None = None
  ) -> list[dict[str, Any]]:
    """Forecast points whose target bar has since printed.

    The join against `bars` is what makes a point "resolved": until the bar
    exists the prediction is simply not scored.
    """
    query = (
      "SELECT f.origin_ts, f.origin_close, p.step, p.target_ts, p.median,"
      "       p.q10, p.q90, b.close AS actual"
      "  FROM forecast_points p"
      "  JOIN forecasts f ON f.id = p.forecast_id"
      "  JOIN bars b ON b.ts = p.target_ts"
      " WHERE f.config_id = ?"
    )
    params: list[Any] = [config_id]
    if step is not None:
      query += " AND p.step = ?"
      params.append(step)
    query += " ORDER BY p.target_ts DESC, p.step LIMIT ?"
    params.append(limit)
    with self._lock:
      rows = self._conn.execute(query, params).fetchall()
    return [dict(r) for r in reversed(rows)]

  def pending_count(self, config_id: str) -> int:
    """Points still waiting for their bar to print."""
    with self._lock:
      row = self._conn.execute(
        "SELECT COUNT(*) AS n FROM forecast_points p"
        "  JOIN forecasts f ON f.id = p.forecast_id"
        "  LEFT JOIN bars b ON b.ts = p.target_ts"
        " WHERE f.config_id = ? AND b.ts IS NULL",
        (config_id,),
      ).fetchone()
    return int(row["n"])

  def forecast_count(self, config_id: str) -> int:
    with self._lock:
      row = self._conn.execute(
        "SELECT COUNT(*) AS n FROM forecasts WHERE config_id = ?", (config_id,)
      ).fetchone()
    return int(row["n"])


# --- Scoring ----------------------------------------------------------------


def score(points: list[dict[str, Any]]) -> dict[str, Any]:
  """Aggregates resolved points into the metrics the dashboard ranks on.

  `persistence_mae` is the error of the trivial "price does not move" forecast
  over the exact same points. `skill` below 1 means the model beat it. For
  intraday crypto that is a demanding reference, so it is reported next to every
  headline number rather than buried.
  """
  if not points:
    return {
      "n": 0,
      "mae": None,
      "rmse": None,
      "mape": None,
      "direction_acc": None,
      "coverage_80": None,
      "persistence_mae": None,
      "skill": None,
      "bias": None,
    }

  predicted = np.array([p["median"] for p in points], dtype=np.float64)
  actual = np.array([p["actual"] for p in points], dtype=np.float64)
  origin = np.array([p["origin_close"] for p in points], dtype=np.float64)
  q_low = np.array([p["q10"] for p in points], dtype=np.float64)
  q_high = np.array([p["q90"] for p in points], dtype=np.float64)

  error = predicted - actual
  mae = float(np.mean(np.abs(error)))
  rmse = float(np.sqrt(np.mean(error**2)))
  with np.errstate(divide="ignore", invalid="ignore"):
    mape = float(np.mean(np.abs(error / np.where(actual != 0, actual, np.nan))) * 100.0)

  predicted_move = predicted - origin
  actual_move = actual - origin
  # Points where the market did not move at all carry no directional
  # information; excluding them keeps the hit rate honest.
  moved = actual_move != 0
  direction_acc = (
    float(np.mean(np.sign(predicted_move[moved]) == np.sign(actual_move[moved])) * 100.0)
    if moved.any()
    else None
  )

  persistence_mae = float(np.mean(np.abs(origin - actual)))

  return {
    "n": len(points),
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "direction_acc": direction_acc,
    "coverage_80": float(np.mean((actual >= q_low) & (actual <= q_high)) * 100.0),
    "persistence_mae": persistence_mae,
    "skill": (mae / persistence_mae) if persistence_mae > 0 else None,
    "bias": float(np.mean(error)),
  }
