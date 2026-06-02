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
                features.append(float(log_ret[-1]))
                features.append(float(np.mean(log_ret)))
                features.append(float(np.std(log_ret) + 1e-8))
            except Exception:
                features.extend([0.0, 0.0, 1e-8])

        return np.array(features, dtype=np.float32)
