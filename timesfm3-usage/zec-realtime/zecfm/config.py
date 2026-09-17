"""Static configuration for the ZEC-USDT realtime forecasting service.

Everything that defines the experiment -- the bar size, the context window, the
13 covariate channels and the three model configurations -- lives here so the
backend, the dashboard and the tests all read the same definitions.

The 5.9 hour context is not an arbitrary choice. OKX's taker buy/sell volume
endpoint (`/api/v5/rubik/stat/taker-volume`) only retains 72 five-minute
buckets, and the newest of those is always still in progress. That leaves 71
*closed* buckets = 355 minutes = 5.9167 hours of usable history, and it is the
shortest history of any of the 13 channels. Every configuration is pinned to
that same window so the three of them stay directly comparable.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

# --- Instrument -------------------------------------------------------------

INST_ID = "ZEC-USDT-SWAP"  # OKX perpetual swap, the instrument we forecast
CCY = "ZEC"  # currency code used by the rubik (derivatives statistics) endpoints
BAR = "5m"  # OKX bar size
BAR_MS = 5 * 60 * 1000

# --- Window -----------------------------------------------------------------

#: Closed 5m bars per context window. 71 * 5min = 355min = 5.9167h ~ "5.9 hours",
#: which is exactly what the taker-volume endpoint can supply.
CONTEXT_BARS = 71
CONTEXT_HOURS = CONTEXT_BARS * BAR_MS / 3_600_000  # 5.9166...

#: Forecast horizon in bars. 12 * 5min = 60 minutes ahead.
HORIZON_BARS = 12

#: Extra bars fetched before the context window purely to warm up the technical
#: indicators (MACD needs 26+9 bars, RSI 14, and so on). They are dropped before
#: anything is handed to the model, so no indicator inside the context window is
#: ever computed from a partial lookback.
WARMUP_BARS = 80

#: Candles requested per poll. OKX caps `/market/candles` at 300.
CANDLE_FETCH_LIMIT = min(300, CONTEXT_BARS + WARMUP_BARS + 20)

# --- Channels ---------------------------------------------------------------

ChannelGroup = Literal["technical", "derivative"]


@dataclasses.dataclass(frozen=True)
class Channel:
  """One covariate channel fed to TimesFM as a past-only covariate."""

  key: str
  label_zh: str
  label_en: str
  group: ChannelGroup
  unit: str
  #: Where the numbers come from, shown in the dashboard's channel panel.
  source: str
  #: Short description of how the channel is derived.
  detail: str


TECHNICAL_CHANNELS: tuple[Channel, ...] = (
  Channel(
    key="rsi_14",
    label_zh="RSI",
    label_en="RSI(14)",
    group="technical",
    unit="0-100",
    source="5m K线收盘价",
    detail="Wilder RSI, 14 根 5 分钟 K 线",
  ),
  Channel(
    key="macd_hist",
    label_zh="MACD",
    label_en="MACD histogram",
    group="technical",
    unit="% of price",
    source="5m K线收盘价",
    detail="(EMA12-EMA26) - DEA9，除以收盘价归一为百分比",
  ),
  Channel(
    key="ema_bias",
    label_zh="EMA 乖离",
    label_en="EMA bias",
    group="technical",
    unit="%",
    source="5m K线收盘价",
    detail="(close - EMA20) / EMA20 * 100",
  ),
  Channel(
    key="atr_pct",
    label_zh="ATR",
    label_en="ATR(14) %",
    group="technical",
    unit="%",
    source="5m K线 OHLC",
    detail="Wilder ATR(14) 除以收盘价，转百分比后与价格尺度解耦",
  ),
  Channel(
    key="mom_15m",
    label_zh="15 分钟动量",
    label_en="15m momentum",
    group="technical",
    unit="%",
    source="5m K线收盘价",
    detail="close_t / close_{t-3} - 1，3 根 5 分钟 K 线 = 15 分钟",
  ),
  Channel(
    key="mom_60m",
    label_zh="60 分钟动量",
    label_en="60m momentum",
    group="technical",
    unit="%",
    source="5m K线收盘价",
    detail="close_t / close_{t-12} - 1，12 根 5 分钟 K 线 = 60 分钟",
  ),
  Channel(
    key="realized_vol",
    label_zh="已实现波动率",
    label_en="Realized volatility",
    group="technical",
    unit="%/h",
    source="5m K线收盘价",
    detail="12 根 K 线对数收益率标准差 × sqrt(12)，折算为小时波动率",
  ),
  Channel(
    key="volume_ratio",
    label_zh="量比",
    label_en="Volume ratio",
    group="technical",
    unit="x",
    source="5m K线成交额",
    detail="当根成交额 / 前 12 根成交额均值，1.0 表示与近一小时均量持平",
  ),
)

DERIVATIVE_CHANNELS: tuple[Channel, ...] = (
  Channel(
    key="funding_rate",
    label_zh="资金费率",
    label_en="Funding rate",
    group="derivative",
    unit="bp",
    source="/api/v5/public/funding-rate(-history)",
    detail="当期资金费率，按基点(1bp=0.01%)给出；服务运行期间实时采样，冷启动回落到 8 小时结算历史的阶梯值",
  ),
  Channel(
    key="taker_buy_share",
    label_zh="主动买卖占比",
    label_en="Taker buy share",
    group="derivative",
    unit="%",
    source="/api/v5/rubik/stat/taker-volume",
    detail="主动买入量 / (主动买入量 + 主动卖出量)，>50 表示主动买盘占优。本通道只保留 72 个 5 分钟桶，是 5.9 小时窗口的约束来源",
  ),
  Channel(
    key="long_short_account_ratio",
    label_zh="多空人数比",
    label_en="Long/short account ratio",
    group="derivative",
    unit="ratio",
    source="/api/v5/rubik/stat/contracts/long-short-account-ratio",
    detail="持多头仓位人数 / 持空头仓位人数",
  ),
  Channel(
    key="margin_loan_ratio",
    label_zh="杠杆多空比",
    label_en="Margin long/short ratio",
    group="derivative",
    unit="ratio",
    source="/api/v5/rubik/stat/margin/loan-ratio",
    detail="杠杆借贷比：计价币借币量 / 基础币借币量",
  ),
  Channel(
    key="open_interest",
    label_zh="合约持仓量",
    label_en="Open interest",
    group="derivative",
    unit="M USD",
    source="/api/v5/rubik/stat/contracts/open-interest-volume",
    detail="全市场 ZEC 合约持仓名义价值，单位百万美元",
  ),
)

ALL_CHANNELS: tuple[Channel, ...] = TECHNICAL_CHANNELS + DERIVATIVE_CHANNELS
CHANNELS_BY_KEY = {c.key: c for c in ALL_CHANNELS}

TECHNICAL_KEYS = tuple(c.key for c in TECHNICAL_CHANNELS)
DERIVATIVE_KEYS = tuple(c.key for c in DERIVATIVE_CHANNELS)
ALL_KEYS = TECHNICAL_KEYS + DERIVATIVE_KEYS

assert len(TECHNICAL_KEYS) == 8, "spec fixes the technical block at 8 channels"
assert len(DERIVATIVE_KEYS) == 5, "spec fixes the derivative block at 5 channels"

# --- Model configurations ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ForecastConfig:
  """One of the three side-by-side model configurations.

  All three share the same target (the 5m close), the same 5.9h context and the
  same horizon; they differ only in which covariate channels they see. That
  makes the accuracy table an ablation: config B minus A isolates what the
  technical block buys, C minus B what the derivatives block adds.
  """

  id: str
  name_zh: str
  name_en: str
  channels: tuple[str, ...]
  color: str
  blurb_zh: str

  @property
  def num_covariates(self) -> int:
    return len(self.channels)


CONFIGS: tuple[ForecastConfig, ...] = (
  ForecastConfig(
    id="price_only",
    name_zh="A · 纯价格",
    name_en="A - price only",
    channels=(),
    color="#94a3b8",
    blurb_zh="零协变量基线：只喂 5.9 小时收盘价，用来衡量协变量到底有没有带来增益。",
  ),
  ForecastConfig(
    id="technical",
    name_zh="B · 价格 + 8 技术指标",
    name_en="B - price + 8 technicals",
    channels=TECHNICAL_KEYS,
    color="#38bdf8",
    blurb_zh="只用 K 线自身可以算出的 8 个技术指标，不接触任何衍生品数据。",
  ),
  ForecastConfig(
    id="full",
    name_zh="C · 价格 + 13 全通道",
    name_en="C - price + all 13",
    channels=ALL_KEYS,
    color="#f472b6",
    blurb_zh="8 技术指标 + 5 衍生品通道，完整配置。",
  ),
)

CONFIGS_BY_ID = {c.id: c for c in CONFIGS}

# --- Service defaults -------------------------------------------------------

#: How often the poller refreshes market data (seconds). A fresh forecast is
#: only produced when a new 5m bar closes, but actuals/derivatives are refreshed
#: on this cadence so the dashboard stays live.
POLL_SECONDS = 20

#: Quantile levels emitted by TimesFM 3.0, in order.
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: Nominal coverage of the band drawn on the dashboard (q10..q90).
BAND_LOW_Q, BAND_HIGH_Q = 0.1, 0.9
