import numpy as np
import pandas as pd


class FeatureHandler:
    """
    Converts market data into a fixed-length state vector for the PPO model.

    For each asset computes 3 features over a rolling window:
      - latest log return
      - mean log return (momentum proxy)
      - std of log returns (volatility proxy)

    state_dim = len(assets) * 3
    """

    def __init__(self, data_handler, assets, lookback=20):
        self.data_handler = data_handler
        self.assets = assets
        self.lookback = lookback
        # Fallback statistics computed from the first full window seen.
        # Populated lazily; avoids all-zero observations during warm-up.
        self._fallback_mean = {}
        self._fallback_std  = {}

    def __call__(self, dt):
        # Need lookback+1 closes to compute lookback returns; add buffer for holidays
        start_dt = pd.Timestamp(dt) - pd.tseries.offsets.BDay(self.lookback + 10)
        prices_df = self.data_handler.get_assets_historical_range_close_price(
            start_dt, dt, self.assets
        )

        features = []
        for asset in self.assets:
            try:
                prices = prices_df[asset].dropna().values[-(self.lookback + 1):]
                if len(prices) < 2:
                    raise ValueError("insufficient data")
                log_ret = np.diff(np.log(prices + 1e-8))

                latest_ret  = float(log_ret[-1])
                mean_ret    = float(np.mean(log_ret))
                std_ret     = float(np.std(log_ret) + 1e-8)

                # Update fallback stats as soon as we have a full window
                if len(log_ret) >= self.lookback:
                    self._fallback_mean[asset] = mean_ret
                    self._fallback_std[asset]  = std_ret

                features.append(latest_ret)
                features.append(mean_ret)
                features.append(std_ret)
            except Exception:
                # Use historical fallback if available; otherwise neutral defaults.
                # Neutral default: zero return, small but non-zero volatility proxy.
                features.append(0.0)
                features.append(self._fallback_mean.get(asset, 0.0))
                features.append(self._fallback_std.get(asset, 1e-4))

        return np.array(features, dtype=np.float32)

