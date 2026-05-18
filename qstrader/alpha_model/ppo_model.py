from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.signals.signal import Signal
from stable_baselines3 import PPO
import numpy as np

class PPOModel(AlphaModel):
    def __init__(self, ppo_model_path, assets, feature_handler):
        '''
        feature_handler:将原始数据拼接成state向量
        '''
        self.model = PPO.load(ppo_model_path)
        self.assets = assets
        self.feature_handler = feature_handler

    def construct_signals(self, dt):
        signals = []
        state = self.feature_handler(self.assets, dt)
        action, _ = self.model.predict(state, deterministic=True)
        target_weights = self.action_to_weights(action)
        for i, asset in enumerate(self.assets):
            weight = target_weights[i]
            signal = Signal(asset, "WEIGHT", weight, dt)
            signals.append(signal)
        return signals
    
    def action_to_weights(self, action):
        exp_action = np.exp(action - np.max(action))
        return exp_action / np.sum(exp_action)
