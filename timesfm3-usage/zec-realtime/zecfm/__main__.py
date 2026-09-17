"""CLI entry point: `python -m zecfm [--engine ...] [--port ...]`."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from . import config as cfg
from .app import create_app


def main() -> None:
  parser = argparse.ArgumentParser(
    prog="zecfm",
    description="Realtime ZEC-USDT forecasting with TimesFM 3.0, scored against reality.",
  )
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8000)
  parser.add_argument("--db", default="zecfm.db", help="SQLite path")
  parser.add_argument(
    "--engine",
    default="timesfm",
    choices=["timesfm", "random-walk"],
    help="'random-walk' is a development stand-in, not a forecast",
  )
  parser.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
  parser.add_argument("--device", default=None, help="e.g. cuda, cpu, mps")
  parser.add_argument("--horizon", type=int, default=cfg.HORIZON_BARS, help="bars ahead")
  parser.add_argument("--poll", type=int, default=cfg.POLL_SECONDS, help="seconds")
  parser.add_argument("--log-level", default="info")
  args = parser.parse_args()

  logging.basicConfig(
    level=args.log_level.upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
  )

  app = create_app(
    db_path=args.db,
    engine_kind=args.engine,
    checkpoint_path=args.checkpoint,
    device=args.device,
    horizon_bars=args.horizon,
    poll_seconds=args.poll,
  )
  uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
  main()
