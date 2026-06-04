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


class proxyAlphaModel(AlphaModel):
    def __init__(self, assets, signal_weights):  # Bug1: assets param added; super() takes no args
        super().__init__()
        self.assets = assets
        self.current_actions = signal_weights

    def __call__(self, dt):
        return {                                  # Bug2: return dict[asset->float], not numpy array
            asset: float(w)
            for asset, w in zip(self.assets, self.current_actions)
        }


class QSTraderExecutionEnv(gym.Env):
    def __init__(self, training_config):
        """
        training_config: dict{
            'state_dim':  int,
            'action_dim': int,
            'starting_day': str,
            'ending_day':   str,
            'symbols': list[str],   e.g. ['SPY', 'AGG']
            'assets':  list[str],   e.g. ['EQ:SPY', 'EQ:AGG']
        }

        action_space: [0,1]^action_dim，代表各资产多头权重
        """
        super().__init__()
        self.action_space = spaces.Box(
            low=0, high=1,
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
        self.start_dt = pd.Timestamp(training_config['starting_day'] + ' 14:30:00', tz='UTC')
        self.end_dt   = pd.Timestamp(training_config['ending_day']   + ' 23:59:00', tz='UTC')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        strategy_uni = StaticUniverse(self.assets)
        _default_csv = os.path.join(os.path.dirname(__file__), '..', '..', 'examples')
        csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', _default_csv)
        data_source = CSVDailyBarDataSource(
            csv_dir, Equity, csv_symbols=self.symbols, adjust_prices=False
        )
        data_handler = BacktestDataHandler(strategy_uni, data_sources=[data_source])

        n = len(self.assets)
        self.alpha_model = proxyAlphaModel(
            assets=self.assets,
            signal_weights=np.ones(n) / n
        )

        self.backtest = BacktestTradingSession(
            start_dt=self.start_dt,
            end_dt=self.end_dt,
            universe=strategy_uni,
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
        self.alpha_model.current_actions = action

        next_obs = self._advance_to_market_open()

        portfolio_value = self.backtest.broker.get_account_total_equity()['master']
        reward = float(portfolio_value - self.current_portfolio_value)
        self.current_portfolio_value = portfolio_value

        terminated = next_obs is None
        truncated = False
        if terminated:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return next_obs, reward, terminated, truncated, {}


if __name__ == "__main__":
    env = QSTraderExecutionEnv(training_config={
        'state_dim': 15,           # 5 assets × 3 features (log_ret, mean_ret, std)
        'action_dim': 5,
        'starting_day': '2020-01-01',
        'ending_day': '2020-12-31',
        'symbols': ['SPY', 'AGG', 'GLD', 'IEI', 'TLT'],
        'assets': ['EQ:SPY', 'EQ:AGG', 'EQ:GLD', 'EQ:IEI', 'EQ:TLT']
    })
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Reward: {reward}")
