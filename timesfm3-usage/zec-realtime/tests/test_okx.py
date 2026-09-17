"""Tests for the OKX wire format, against a mock transport.

These pin down the assumptions that would silently corrupt every channel if OKX
ever reordered a row: which column is the buy volume, which is open interest,
and that the in-progress candle is dropped.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from zecfm.okx import OkxClient, OkxError


def _client(handler) -> OkxClient:
  transport = httpx.MockTransport(handler)
  inner = httpx.AsyncClient(base_url="https://okx.test", transport=transport)
  # No real sleeping between retries in tests.
  return OkxClient(client=inner, retry_backoff=0.0)


def _ok(data) -> httpx.Response:
  return httpx.Response(200, text=json.dumps({"code": "0", "msg": "", "data": data}))


def _run(coro):
  return asyncio.run(coro)


async def _with(client: OkxClient, call):
  async with client as c:
    return await call(c)


def test_candles_drop_the_unconfirmed_bar_and_sort_oldest_first():
  rows = [
    # OKX returns newest first; the newest bar here is still forming.
    ["3000", "3", "3.1", "2.9", "3.05", "10", "10", "300", "0"],
    ["2000", "2", "2.1", "1.9", "2.05", "10", "10", "200", "1"],
    ["1000", "1", "1.1", "0.9", "1.05", "10", "10", "100", "1"],
  ]
  candles = _run(_with(_client(lambda r: _ok(rows)), lambda c: c.fetch_candles()))
  assert [c.ts for c in candles] == [1000, 2000]
  assert candles[-1].close == pytest.approx(2.05)
  # volCcyQuote (index 7) is the volume the 量比 channel is built from.
  assert candles[-1].volume_quote == pytest.approx(200.0)


def test_taker_volume_row_is_sell_then_buy():
  # [ts, sellVol, buyVol] -> a buy share of 75%.
  rows = [["1000", "25", "75"]]
  out = _run(_with(_client(lambda r: _ok(rows)), lambda c: c.fetch_taker_buy_share()))
  assert out[0].value == pytest.approx(75.0)


def test_taker_volume_skips_empty_buckets_rather_than_dividing_by_zero():
  rows = [["1000", "0", "0"], ["2000", "50", "50"]]
  out = _run(_with(_client(lambda r: _ok(rows)), lambda c: c.fetch_taker_buy_share()))
  assert [s.ts for s in out] == [2000]


def test_open_interest_reads_the_first_value_and_scales_to_millions():
  # [ts, oi, vol] -- the second column is volume and must not be picked up.
  rows = [["1000", "207000000", "15000000"]]
  out = _run(_with(_client(lambda r: _ok(rows)), lambda c: c.fetch_open_interest()))
  assert out[0].value == pytest.approx(207.0)


def test_ratio_endpoints_parse_and_sort():
  rows = [["2000", "0.52"], ["1000", "0.34"]]
  out = _run(
    _with(_client(lambda r: _ok(rows)), lambda c: c.fetch_long_short_account_ratio())
  )
  assert [(s.ts, s.value) for s in out] == [(1000, 0.34), (2000, 0.52)]


def test_funding_rate_is_converted_to_basis_points():
  rows = [{"fundingRate": "-0.0002883809", "ts": "1789646549667"}]
  rate, ts = _run(_with(_client(lambda r: _ok(rows)), lambda c: c.fetch_funding_now()))
  assert rate == pytest.approx(-2.883809, rel=1e-6)
  assert ts == 1789646549667


def test_funding_history_is_converted_and_sorted():
  rows = [
    {"fundingTime": "2000", "fundingRate": "0.0001"},
    {"fundingTime": "1000", "fundingRate": "-0.0002"},
  ]
  out = _run(
    _with(_client(lambda r: _ok(rows)), lambda c: c.fetch_funding_history())
  )
  assert [(s.ts, round(s.value, 6)) for s in out] == [(1000, -2.0), (2000, 1.0)]


def test_business_error_code_is_raised():
  def handler(request):
    return httpx.Response(200, text=json.dumps({"code": "51001", "msg": "nope", "data": []}))

  with pytest.raises(OkxError, match="51001"):
    _run(_with(_client(handler), lambda c: c.fetch_candles()))


def test_transient_status_is_retried_then_succeeds():
  calls = {"n": 0}

  def handler(request):
    calls["n"] += 1
    if calls["n"] < 3:
      return httpx.Response(503, text="busy")
    return _ok([["1000", "25", "75"]])

  client = _client(handler)
  client._max_retries = 3
  out = _run(_with(client, lambda c: c.fetch_taker_buy_share()))
  assert calls["n"] == 3
  assert out[0].value == pytest.approx(75.0)


def test_retries_are_bounded():
  def handler(request):
    return httpx.Response(503, text="busy")

  client = _client(handler)
  client._max_retries = 2
  with pytest.raises(OkxError, match="failed after 2 attempts"):
    _run(_with(client, lambda c: c.fetch_taker_buy_share()))


def test_snapshot_gathers_every_series():
  def handler(request):
    path = request.url.path
    if path.endswith("/candles"):
      return _ok([["1000", "1", "1.1", "0.9", "1.05", "1", "1", "100", "1"]])
    if path.endswith("/funding-rate"):
      return _ok([{"fundingRate": "0.0001", "ts": "1000"}])
    if path.endswith("/funding-rate-history"):
      return _ok([{"fundingTime": "1000", "fundingRate": "0.0001"}])
    if path.endswith("/taker-volume"):
      return _ok([["1000", "25", "75"]])
    if path.endswith("/open-interest-volume"):
      return _ok([["1000", "1000000", "1"]])
    return _ok([["1000", "0.5"]])

  snap = _run(_with(_client(handler), lambda c: c.fetch_snapshot()))
  assert len(snap.candles) == 1
  assert snap.taker_buy_share[0].value == pytest.approx(75.0)
  assert snap.open_interest[0].value == pytest.approx(1.0)
  assert snap.long_short_account_ratio[0].value == pytest.approx(0.5)
  assert snap.margin_loan_ratio[0].value == pytest.approx(0.5)
  assert snap.funding_now_bp == pytest.approx(1.0)
