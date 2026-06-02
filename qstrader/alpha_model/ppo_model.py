import numpy as np
from stable_baselines3 import PPO

from qstrader.alpha_model.alpha_model import AlphaModel


class PPOModel(AlphaModel):
    def __init__(self, ppo_model_path, assets, feature_handler):
        """
        ppo_model_path  : path to a saved stable-baselines3 PPO .zip file
        assets          : list of asset strings, e.g. ['EQ:SPY', 'EQ:AGG']
        feature_handler : FeatureHandler(dt) -> np.ndarray of shape (state_dim,)
        """
        self.model = PPO.load(ppo_model_path)
        self.assets = assets
        self.feature_handler = feature_handler

    def __call__(self, dt):
        state = self.feature_handler(dt)          # only dt; handler owns data_handler/assets
        action, _ = self.model.predict(state, deterministic=True)
        weights = self._softmax(action)
        return {asset: float(w) for asset, w in zip(self.assets, weights)}

    def _softmax(self, action):
        exp = np.exp(action - np.max(action))
        return exp / np.sum(exp)
