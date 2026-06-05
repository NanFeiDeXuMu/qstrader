import numpy as np


class BacktestDataHandler(object):
    """
    """

    def __init__(
        self,
        universe,
        data_sources=None
    ):
        self.universe = universe
        self.data_sources = data_sources
        self.cumulative_offsets = {}

    def get_asset_latest_bid_price(self, dt, asset_symbol):
        """
        """
        # TODO: Check for asset in Universe
        bid = np.nan
        for ds in self.data_sources:
            try:
                bid = ds.get_bid(dt, asset_symbol)
                if not np.isnan(bid):
                    return bid
            except Exception:
                bid = np.nan
        bid += self.cumulative_offsets.get(asset_symbol, 0.0)
        return bid

    def get_asset_latest_ask_price(self, dt, asset_symbol):
        """
        """
        # TODO: Check for asset in Universe
        ask = np.nan
        for ds in self.data_sources:
            try:
                ask = ds.get_ask(dt, asset_symbol)
                if not np.isnan(ask):
                    return ask
            except Exception:
                ask = np.nan
        ask += self.cumulative_offsets.get(asset_symbol, 0.0)
        return ask

    def get_asset_latest_bid_ask_price(self, dt, asset_symbol):
        """
        """
        # TODO: For the moment this is sufficient for OHLCV
        # data, which only usually provides mid prices
        # This will need to be revisited when handling intraday
        # bid/ask time series.
        # It has been added as an optimisation mechanism for
        # interday backtests.
        bid = self.get_asset_latest_bid_price(dt, asset_symbol)
        ask = self.get_asset_latest_ask_price(dt, asset_symbol)
        return (bid, ask)

    def get_asset_latest_mid_price(self, dt, asset_symbol):
        """
        """
        bid_ask = self.get_asset_latest_bid_ask_price(dt, asset_symbol)
        try:
            mid = (bid_ask[0] + bid_ask[1]) / 2.0
        except Exception:
            # TODO: Log this
            mid = np.nan
        # Do NOT add cumulative_offsets here: bid/ask getters already return
        # the raw CSV price when valid data exists (early return, no offset),
        # so adding offset here would inflate mid prices and create a
        # runaway feedback loop in position marking (total_equity → inf).
        return mid

    def get_assets_historical_range_close_price(
        self, start_dt, end_dt, asset_symbols, adjusted=False
    ):
        """
        """
        prices_df = None
        for ds in self.data_sources:
            try:
                prices_df = ds.get_assets_historical_closes(
                    start_dt, end_dt, asset_symbols
                )
                if prices_df is not None:
                    return prices_df
            except Exception:
                raise
        return prices_df
    
    def set_last_price(self, asset, price_change, dt=0.0):
        self.cumulative_offsets[asset] = self.cumulative_offsets.get(asset, 0.0) + price_change
        print(f"For {asset}, total offset is {self.cumulative_offsets[asset]}")
