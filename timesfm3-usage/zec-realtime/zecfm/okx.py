"""Async OKX v5 REST client for every series the forecaster needs.

Only public market-data endpoints are used, so no API key is required. All of
them return newest-first arrays; this module flips them to oldest-first and
converts the string fields OKX sends into floats, so nothing downstream has to
think about wire format.

Endpoint map (this is the whole data surface of the project):

  5m OHLCV               /api/v5/market/candles
  funding rate (live)    /api/v5/public/funding-rate
  funding rate (settled) /api/v5/public/funding-rate-history
  taker buy/sell volume  /api/v5/rubik/stat/taker-volume
  long/short accounts    /api/v5/rubik/stat/contracts/long-short-account-ratio
  margin long/short      /api/v5/rubik/stat/margin/loan-ratio
  open interest          /api/v5/rubik/stat/contracts/open-interest-volume

Note that the four `rubik` endpoints are keyed by currency (`ccy=ZEC`), not by
instrument: they aggregate ZEC contracts across the venue rather than describing
ZEC-USDT-SWAP alone. That is how OKX publishes them, and it is what the
dashboard's channel panel says the numbers are.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import Any

import httpx

from . import config as cfg

_LOG = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://www.okx.com"

#: Endpoints occasionally answer with a transient 50011 (rate limit) or a 5xx.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class OkxError(RuntimeError):
  """Raised when OKX answers with a non-zero business code."""


@dataclasses.dataclass(frozen=True)
class Candle:
  """One closed 5-minute candle."""

  ts: int  # bucket open time, ms
  open: float
  high: float
  low: float
  close: float
  volume_quote: float  # traded value in USDT (volCcyQuote)


@dataclasses.dataclass(frozen=True)
class Sample:
  """A generic (timestamp, value) point from a rubik statistics endpoint."""

  ts: int
  value: float


@dataclasses.dataclass(frozen=True)
class MarketSnapshot:
  """Everything fetched in a single poll cycle."""

  fetched_ms: int
  candles: list[Candle]
  taker_buy_share: list[Sample]
  long_short_account_ratio: list[Sample]
  margin_loan_ratio: list[Sample]
  open_interest: list[Sample]
  funding_now_bp: float | None
  funding_now_ts: int | None
  funding_history_bp: list[Sample]


def _f(value: str) -> float:
  return float(value)


class OkxClient:
  """Thin async wrapper over the handful of endpoints we need."""

  def __init__(
    self,
    base_url: str = DEFAULT_BASE_URL,
    *,
    timeout: float = 15.0,
    max_retries: int = 3,
    retry_backoff: float = 1.0,
    client: httpx.AsyncClient | None = None,
  ) -> None:
    self._base_url = base_url.rstrip("/")
    self._timeout = timeout
    self._max_retries = max_retries
    #: Multiplier on the exponential backoff; tests set it to 0 to run instantly.
    self._retry_backoff = retry_backoff
    self._client = client
    self._owns_client = client is None

  async def __aenter__(self) -> "OkxClient":
    if self._client is None:
      self._client = httpx.AsyncClient(
        base_url=self._base_url,
        timeout=self._timeout,
        headers={"User-Agent": "timesfm3-zec-realtime/1.0"},
      )
    return self

  async def __aexit__(self, *exc_info) -> None:
    if self._owns_client and self._client is not None:
      await self._client.aclose()
      self._client = None

  async def _get(self, path: str, params: dict[str, str | int]) -> list[Any]:
    """GETs an OKX endpoint and returns its `data` array, retrying transients.

    Every endpoint goes through here, including the ones whose rows are objects
    rather than arrays, so a transient failure on any single series is retried
    instead of failing the whole poll cycle.
    """
    if self._client is None:
      raise RuntimeError("OkxClient must be used as an async context manager")

    last_error: Exception | None = None
    for attempt in range(self._max_retries):
      if attempt:
        delay = min(2**attempt, 8) * self._retry_backoff
        if delay > 0:
          await asyncio.sleep(delay)
      try:
        response = await self._client.get(path, params=params)
      except httpx.HTTPError as exc:  # network-level failure
        last_error = exc
        _LOG.warning("OKX %s network error (attempt %d): %s", path, attempt + 1, exc)
        continue

      if response.status_code in _RETRY_STATUS:
        last_error = OkxError(f"{path} -> HTTP {response.status_code}")
        _LOG.warning("OKX %s HTTP %d (attempt %d)", path, response.status_code, attempt + 1)
        continue
      if response.status_code != 200:
        raise OkxError(f"{path} -> HTTP {response.status_code}: {response.text[:200]}")

      payload = response.json()
      if payload.get("code") != "0":
        raise OkxError(f"{path} -> code {payload.get('code')}: {payload.get('msg')}")
      return payload.get("data") or []

    raise OkxError(f"{path} failed after {self._max_retries} attempts") from last_error

  # --- Market data ----------------------------------------------------------

  async def fetch_candles(
    self,
    inst_id: str = cfg.INST_ID,
    bar: str = cfg.BAR,
    limit: int = cfg.CANDLE_FETCH_LIMIT,
  ) -> list[Candle]:
    """Returns closed candles, oldest first.

    OKX marks the in-progress bucket with `confirm == "0"`; it is dropped here so
    no caller ever sees a partial bar. Row layout is
    [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm].
    """
    rows = await self._get(
      "/api/v5/market/candles",
      {"instId": inst_id, "bar": bar, "limit": limit},
    )
    candles = [
      Candle(
        ts=int(r[0]),
        open=_f(r[1]),
        high=_f(r[2]),
        low=_f(r[3]),
        close=_f(r[4]),
        volume_quote=_f(r[7]),
      )
      for r in rows
      if r[8] == "1"
    ]
    candles.sort(key=lambda c: c.ts)
    return candles

  async def fetch_ticker(self, inst_id: str = cfg.INST_ID) -> dict[str, float]:
    """Last price plus the 24h open, for the dashboard header."""
    rows = await self._get("/api/v5/market/ticker", {"instId": inst_id})
    if not rows:
      raise OkxError("ticker returned no rows")
    row = rows[0]
    return {
      "last": _f(row["last"]),
      "open_24h": _f(row["open24h"]),
      "high_24h": _f(row["high24h"]),
      "low_24h": _f(row["low24h"]),
      "vol_24h_quote": _f(row["volCcy24h"]),
      "ts": int(row["ts"]),
    }

  # --- Derivative statistics ------------------------------------------------

  async def fetch_taker_buy_share(
    self, ccy: str = cfg.CCY, period: str = cfg.BAR
  ) -> list[Sample]:
    """Taker buy share in percent, oldest first.

    Row layout is [ts, sellVol, buyVol] -- sell first, per OKX's documented
    response order. The channel is expressed as buy / (buy + sell) * 100 so it
    is bounded in [0, 100] and reads as "percent of aggressive flow that lifted
    the offer".

    This endpoint retains only 72 five-minute buckets, which is what pins every
    configuration to a 5.9 hour context.
    """
    rows = await self._get(
      "/api/v5/rubik/stat/taker-volume",
      {"ccy": ccy, "instType": "CONTRACTS", "period": period},
    )
    out: list[Sample] = []
    for r in rows:
      sell, buy = _f(r[1]), _f(r[2])
      total = sell + buy
      if total <= 0:
        continue
      out.append(Sample(ts=int(r[0]), value=buy / total * 100.0))
    out.sort(key=lambda s: s.ts)
    return out

  async def fetch_long_short_account_ratio(
    self, ccy: str = cfg.CCY, period: str = cfg.BAR
  ) -> list[Sample]:
    """Ratio of accounts holding longs to accounts holding shorts."""
    rows = await self._get(
      "/api/v5/rubik/stat/contracts/long-short-account-ratio",
      {"ccy": ccy, "period": period},
    )
    out = [Sample(ts=int(r[0]), value=_f(r[1])) for r in rows]
    out.sort(key=lambda s: s.ts)
    return out

  async def fetch_margin_loan_ratio(
    self, ccy: str = cfg.CCY, period: str = cfg.BAR
  ) -> list[Sample]:
    """Margin lending ratio: quote-currency borrowing over base-currency borrowing."""
    rows = await self._get(
      "/api/v5/rubik/stat/margin/loan-ratio",
      {"ccy": ccy, "period": period},
    )
    out = [Sample(ts=int(r[0]), value=_f(r[1])) for r in rows]
    out.sort(key=lambda s: s.ts)
    return out

  async def fetch_open_interest(
    self, ccy: str = cfg.CCY, period: str = cfg.BAR
  ) -> list[Sample]:
    """Contract open interest in millions of USD.

    Row layout is [ts, oi, vol]; the first value is the notional open interest.
    """
    rows = await self._get(
      "/api/v5/rubik/stat/contracts/open-interest-volume",
      {"ccy": ccy, "period": period},
    )
    out = [Sample(ts=int(r[0]), value=_f(r[1]) / 1e6) for r in rows]
    out.sort(key=lambda s: s.ts)
    return out

  async def fetch_funding_now(self, inst_id: str = cfg.INST_ID) -> tuple[float, int]:
    """Current-period funding rate in basis points, with its observation time."""
    rows = await self._get("/api/v5/public/funding-rate", {"instId": inst_id})
    if not rows:
      raise OkxError("funding-rate returned no rows")
    row = rows[0]
    return _f(row["fundingRate"]) * 10_000.0, int(row["ts"])

  async def fetch_funding_history(
    self, inst_id: str = cfg.INST_ID, limit: int = 10
  ) -> list[Sample]:
    """Settled funding rates in basis points, oldest first.

    Funding settles every 8 hours, so within a 5.9 hour window this is a step
    function with at most one step. It is the cold-start fallback for the
    funding channel; once the service has been running, live samples recorded
    each poll give the channel real intra-period movement.
    """
    rows = await self._get(
      "/api/v5/public/funding-rate-history", {"instId": inst_id, "limit": limit}
    )
    out = [
      Sample(ts=int(r["fundingTime"]), value=_f(r["fundingRate"]) * 10_000.0)
      for r in rows
    ]
    out.sort(key=lambda s: s.ts)
    return out

  # --- Composite ------------------------------------------------------------

  async def fetch_snapshot(self) -> MarketSnapshot:
    """Fetches every series for one poll cycle concurrently."""
    (
      candles,
      taker,
      ls_accounts,
      margin,
      open_interest,
      funding_now,
      funding_hist,
    ) = await asyncio.gather(
      self.fetch_candles(),
      self.fetch_taker_buy_share(),
      self.fetch_long_short_account_ratio(),
      self.fetch_margin_loan_ratio(),
      self.fetch_open_interest(),
      self.fetch_funding_now(),
      self.fetch_funding_history(),
    )
    return MarketSnapshot(
      fetched_ms=int(time.time() * 1000),
      candles=candles,
      taker_buy_share=taker,
      long_short_account_ratio=ls_accounts,
      margin_loan_ratio=margin,
      open_interest=open_interest,
      funding_now_bp=funding_now[0],
      funding_now_ts=funding_now[1],
      funding_history_bp=funding_hist,
    )
