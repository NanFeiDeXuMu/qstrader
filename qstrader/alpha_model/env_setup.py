import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from qstrader.alpha_model.feature_handler import FeatureHandler
from qstrader.trading.backtest import BacktestTradingSession
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.asset.equity import Equity
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.alpha_model.alpha_model import AlphaModel

# Burn-in: skip this many trading days at episode start so FeatureHandler
# always has a full lookback window and never emits zero-filled observations.
BURN_IN_DAYS = 25
# Length of each randomly-sampled sub-window episode (trading days).
EPISODE_LENGTH_DAYS = 252


class proxyAlphaModel(AlphaModel):
    def __init__(self, assets, signal_weights):
        super().__init__()
        self.assets = assets
        self.current_actions = signal_weights  # raw logits; softmax applied in __call__

    def __call__(self, dt):
        # Softmax: maps unconstrained logits → strictly positive weights summing to 1.
        # Consistent with PPOModel._action_to_weights at inference time.
        logits = np.asarray(self.current_actions, dtype=np.float64)
        logits -= logits.max()          # numerical stability
        exp_w = np.exp(logits)
        weights = exp_w / exp_w.sum()
        return {asset: float(w) for asset, w in zip(self.assets, weights)}


class QSTraderExecutionEnv(gym.Env):
    def __init__(self, training_config):
        """
        training_config: dict{
            'state_dim':  int,
            'action_dim': int,
            'starting_day': str,   # earliest possible episode start date
            'ending_day':   str,   # latest possible episode end date
            'symbols': list[str],
            'assets':  list[str],
            'episode_length_days': int  (optional, default EPISODE_LENGTH_DAYS)
        }

        action_space: unconstrained R^action_dim logits; softmax gives portfolio weights.
        Using Box(-inf, inf) avoids the gradient-signal degeneracy of the old [0,1] box
        where many distinct raw actions mapped to the same normalised weight vector.
        """
        super().__init__()
        # Logit action space — softmax applied inside proxyAlphaModel.
        # SB3 ≥2.3 requires finite bounds; ±1e4 is functionally unbounded for
        # softmax (differences >~10 already saturate to near-one-hot weights).
        self.action_space = spaces.Box(
            low=-1e4, high=1e4,
            shape=(training_config['action_dim'],),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(training_config['state_dim'],),
            dtype=np.float32
        )
        self.state_dim = training_config['state_dim']
        self.symbols = training_config['symbols']
        self.assets = training_config['assets']
        self.episode_length_days = training_config.get(
            'episode_length_days', EPISODE_LENGTH_DAYS
        )

        # Full date range for random window sampling
        self._range_start = pd.Timestamp(
            training_config['starting_day'] + ' 00:00:00', tz='UTC'
        )
        self._range_end = pd.Timestamp(
            training_config['ending_day'] + ' 23:59:00', tz='UTC'
        )

        # Build the data source ONCE here and reuse across resets to avoid
        # repeatedly allocating large LRU-cached DataFrames that leak memory.
        _default_csv = os.path.join(os.path.dirname(__file__), '..', '..', 'examples')
        csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', _default_csv)
        self._strategy_uni = StaticUniverse(self.assets)
        self._data_source = CSVDailyBarDataSource(
            csv_dir, Equity, csv_symbols=self.symbols, adjust_prices=False
        )
        # Pre-compute the sorted list of valid business days in the full range
        # so reset() can cheaply sample a random sub-window start.
        all_bdays = pd.bdate_range(self._range_start, self._range_end, freq='B')
        # Keep only days where price data actually exists (drop leading NaNs)
        first_price_day = all_bdays[BURN_IN_DAYS] if len(all_bdays) > BURN_IN_DAYS else all_bdays[0]
        # Latest start that still leaves a full episode within the range
        latest_start = self._range_end - pd.tseries.offsets.BDay(self.episode_length_days + 1)
        valid_starts = all_bdays[
            (all_bdays >= first_price_day) & (all_bdays <= latest_start)
        ]
        self._valid_starts = valid_starts
        if len(self._valid_starts) == 0:
            raise ValueError(
                f"No valid episode start dates found between {training_config['starting_day']} "
                f"and {training_config['ending_day']}. The date range must span at least "
                f"{BURN_IN_DAYS + EPISODE_LENGTH_DAYS + 1} business days."
            )

    def _sample_episode_window(self, rng):
        """Pick a random start date; end date is episode_length_days later."""
        idx = rng.integers(0, len(self._valid_starts))
        start_bday = self._valid_starts[idx]
        end_bday = start_bday + pd.tseries.offsets.BDay(self.episode_length_days)
        start_dt = start_bday.normalize().replace(hour=14, minute=30).tz_convert('UTC')
        end_dt   = end_bday.normalize().replace(hour=23, minute=59).tz_convert('UTC')
        return start_dt, end_dt

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        start_dt, end_dt = self._sample_episode_window(rng)

        # Reuse the shared data source; create a fresh data handler each reset
        # (cheap: just a thin wrapper, no CSV re-reads).
        data_handler = BacktestDataHandler(
            self._strategy_uni, data_sources=[self._data_source]
        )

        n = len(self.assets)
        self.alpha_model = proxyAlphaModel(
            assets=self.assets,
            signal_weights=np.zeros(n)  # logits=0 → uniform weights initially
        )

        self.backtest = BacktestTradingSession(
            start_dt=start_dt,
            end_dt=end_dt,
            universe=self._strategy_uni,
            alpha_model=self.alpha_model,
            rebalance='daily',
            long_only=True,
            cash_buffer_percentage=0.01,
            data_handler=data_handler
        )

        self.feature_handler = FeatureHandler(
            data_handler=data_handler,
            assets=self.assets,
            lookback=20
        )

        self.sim_iter = iter(self.backtest.sim_engine)
        self.current_portfolio_value = self.backtest.broker.get_account_total_equity()['master']

        obs = self._advance_to_market_open()
        if obs is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def _advance_to_market_open(self):
        """Consume events until next market_open, executing rebalance at
        each market_close along the way. Returns the observation at
        market_open, or None if the simulation has ended."""
        for event in self.sim_iter:
            dt = event.ts
            self.backtest.broker.update(dt)
            if event.event_type == 'market_close':
                if self.backtest._is_rebalance_event(dt):
                    try:
                        self.backtest.qts(dt, stats={'target_allocations': []})
                    except (ValueError, OverflowError):
                        pass  # invalid price at this bar — keep current positions
                self.backtest._update_equity_curve(dt)
            elif event.event_type == 'market_open':
                return self._get_observation(dt)
        return None

    def _get_observation(self, dt):
        """Return FeatureHandler state vector of shape (state_dim,)."""
        try:
            return self.feature_handler(dt)
        except Exception:
            return np.zeros(self.state_dim, dtype=np.float32)

    def step(self, action):
        self.alpha_model.current_actions = action  # raw logits

        next_obs = self._advance_to_market_open()

        portfolio_value = self.backtest.broker.get_account_total_equity()['master']
        # Log return reward: scale-invariant, stable across different portfolio sizes
        # and market regimes. Avoids the dollar P&L magnitude instability.
        prev = self.current_portfolio_value
        if prev > 0:
            reward = float(np.log(portfolio_value / prev))
        else:
            reward = 0.0
        self.current_portfolio_value = portfolio_value

        terminated = next_obs is None
        truncated = False
        if terminated:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return next_obs, reward, terminated, truncated, {}


if __name__ == "__main__":
    env = QSTraderExecutionEnv(training_config={
        'state_dim': 15,
        'action_dim': 5,
        'starting_day': '2010-01-01',
        'ending_day': '2018-12-31',
        'symbols': ['SPY', 'AGG', 'GLD', 'IEI', 'TLT'],
        'assets': ['EQ:SPY', 'EQ:AGG', 'EQ:GLD', 'EQ:IEI', 'EQ:TLT']
    })
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Reward: {reward:.6f}")
