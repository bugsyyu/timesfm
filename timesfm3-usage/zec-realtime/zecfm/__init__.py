"""Realtime ZEC-USDT forecasting with TimesFM 3.0, scored against reality.

Three configurations (price only / price + 8 technicals / price + all 13
channels) run on the same 5.9 hour context every time a 5-minute bar closes.
Each forecast is stored before the bars it predicts exist, then scored against
them as they print, so the accuracy comparison is out-of-sample by construction.

See `config.py` for the window, the 13 covariate channels and the three
configurations -- it is the single source of truth all other modules read.
"""
