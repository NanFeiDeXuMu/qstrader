import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from feature_handler import FeatureHandler
from qstrader.trading.backtest import BacktestTradingSession
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.asset.equity import Equity
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.alpha_model.alpha_model import AlphaModel

class proxyAlphaModel(AlphaModel):
    def __init__(self, signal_weights):
        super().__init__(signal_weights)
        self.current_actions = signal_weights

    def __call__(self, dt):
        return self.current_actions

class QSTraderExecutionEnv(gym.Env):
    def __init__(self, 
                 training_config):
        """
        training_config:dict{
            'state_dim':int,
            'action_dim':int,
            'starting_day':str,
            'ending_day':str,
            'symbols':list[str],
            'assets': list[str]
        }

        action_space: [0,1]之间,代表投资组合中每个资产的权重,仅多头,不考虑空头和杠杆
        """
        super(QSTraderExecutionEnv, self).__init__()
        self.action_space = spaces.Box(low=0, 
                                       high=1, 
                                       shape=(training_config['action_dim'],),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, 
                                            high=np.inf, 
                                            shape=(training_config['state_dim'],),
                                            dtype=np.float32)
        self.symbols = training_config['symbols']
        self.assets = training_config['assets']
        self.start_dt = training_config['starting_day']
        self.end_dt = training_config['ending_day']
        self.alpha_model = proxyAlphaModel(signal_weights=np.ones(training_config['action_dim']) / training_config['action_dim'])


    def reset(self):
        strategy_uni = StaticUniverse(self.assets)
        csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', '.')
        data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=self.symbols, adjust_prices=False)
        data_handler = BacktestDataHandler(strategy_uni, data_sources=[data_source])
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
        self.events = [ event for event in self.backtest.sim_engine]
        self.event_idx = 0
        self.current_portfolio_value = self.backtest.broker.get_account_total_market_value()


    def step(self, action):
        self.alpha_model.current_actions = action
        next_state, reward, done = self._run_one_step(action)
        return next_state, reward, done, {}
    
    def _run_one_step(self, action):
        event = self.events[self.event_idx]
        while event.event_type != "market_open":
            self.event_idx += 1
            self.backtest.run(external_event=event)
            event = self.events[self.event_idx]
            
        # 计算这个market_open事件的next_state和reward和done 

        next_price = self.backtest.data_handler.get_asset_latest_mid_price(event.dt, self.assets[0])
        portfolio_value = self.backtest.broker.get_account_total_market_value()
        reward = portfolio_value - self.current_portfolio_value
        self.current_portfolio_value = portfolio_value

        done = self.event_idx >= len(self.events) - 1
        return next_price, reward, done


if __name__ == "__main__":
    from qstrader.alpha_model.ppo_model import PPOModel
    ppo_model = PPOModel("ppo_model.zip", assets=None, feature_handler=FeatureHandler())
    env = QSTraderExecutionEnv(training_config={
        'target_quantity': 1000,
        'state_dim': 10,
        'action_dim': 5,
        'starting_day': '2020-01-01',
        'ending_day': '2020-12-31'
    }, alpha_model=ppo_model, pcm=None, broker=None, data_handler=None, portfolio=None)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # 随机动作，实际训练时会用PPO模型预测动作
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}")