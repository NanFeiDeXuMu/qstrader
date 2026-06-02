import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from qstrader.alpha_model.feature_handler import FeatureHandler  # Bug9: absolute import
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
        self.start_dt = training_config['starting_day']
        self.end_dt = training_config['ending_day']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        strategy_uni = StaticUniverse(self.assets)
        csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', '.')
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

        self.sim_iter = iter(self.backtest.sim_engine)  # Bug3: iterator, not exhausted list
        self.current_portfolio_value = self.backtest.broker.get_account_total_market_value()
        self._pending_dt = None

        obs = self._advance_to_market_open()
        if obs is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}  # Bug5: return (obs, info)

    def _advance_to_market_open(self):
        """Consume events until market_open; process each event along the way.
        Returns observation vector at that market_open, or None if simulation ended."""
        for event in self.sim_iter:
            self.backtest.broker.update(event.ts)
            if event.event_type == 'market_close':
                self.backtest._update_equity_curve(event.ts)
            elif event.event_type == 'market_open':  # Bug7: process each event in correct order
                self._pending_dt = event.ts
                return self._get_observation(event.ts)
        return None

    def _get_observation(self, dt):
        """Return observation of shape (state_dim,)."""  # Bug4: vector, not scalar
        obs = np.zeros(self.state_dim, dtype=np.float32)
        for i, asset in enumerate(self.assets[:self.state_dim]):
            try:
                price = self.backtest.data_handler.get_asset_latest_mid_price(dt, asset)
                obs[i] = float(price)
            except Exception:
                pass
        return obs

    def step(self, action):
        self.alpha_model.current_actions = action

        # Execute rebalance at the pending market_open with updated weights
        stats = {'target_allocations': []}
        if self.backtest._is_rebalance_event(self._pending_dt):
            self.backtest.qts(self._pending_dt, stats=stats)

        # Advance through market_close to next market_open
        next_obs = self._advance_to_market_open()

        portfolio_value = self.backtest.broker.get_account_total_market_value()
        reward = float(portfolio_value - self.current_portfolio_value)
        self.current_portfolio_value = portfolio_value

        terminated = next_obs is None   # Bug8: judge after full event processing
        truncated = False
        if terminated:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return next_obs, reward, terminated, truncated, {}  # Bug6: 5 return values


if __name__ == "__main__":
    env = QSTraderExecutionEnv(training_config={   # Bug10: only training_config param
        'state_dim': 5,
        'action_dim': 5,
        'starting_day': '2020-01-01',
        'ending_day': '2020-12-31',
        'symbols': ['SPY', 'AGG', 'GLD', 'IEI', 'TLT'],           # Bug10: required keys
        'assets': ['EQ:SPY', 'EQ:AGG', 'EQ:GLD', 'EQ:IEI', 'EQ:TLT']
    })
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Reward: {reward}")
