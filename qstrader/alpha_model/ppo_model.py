from qstrader.alpha_model.alpha_model import AlphaModel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import os

class PPOModel(AlphaModel):
    def __init__(self, ppo_model_path, assets, feature_handler,
                 vecnormalize_path=None):
        '''
        ppo_model_path:    path to saved PPO zip
        vecnormalize_path: optional path to ppo_vecnormalize.pkl; when provided,
                           observations are normalised with the same running stats
                           used during training, which is required when VecNormalize
                           was used in ppo_training.py.
        feature_handler:   FeatureHandler instance
        '''
        self.model = PPO.load(ppo_model_path)
        self.assets = assets
        self.feature_handler = feature_handler
        self._vec_norm = None

        if vecnormalize_path and os.path.exists(vecnormalize_path):
            # Load saved normalisation stats (mean/var) for obs-only normalisation.
            # VecNormalize.load requires a dummy env argument; pass None and set
            # training=False so it is used purely as a stateless scaler.
            self._vec_norm = VecNormalize.load(vecnormalize_path, venv=None)
            self._vec_norm.training = False
            self._vec_norm.norm_reward = False

    def __call__(self, dt):
        state = self.feature_handler(dt)
        if self._vec_norm is not None:
            # Normalise obs with training-time running stats before feeding to policy.
            state = self._vec_norm.normalize_obs(state)
        action, _ = self.model.predict(state, deterministic=True)
        weights = self._action_to_weights(action)
        return {asset: float(weights[i]) for i, asset in enumerate(self.assets)}

    def _action_to_weights(self, action):
        # Softmax: matches the proxyAlphaModel used during training.
        # Unconstrained logits → strictly positive weights summing to 1.
        logits = np.asarray(action, dtype=np.float64)
        logits -= logits.max()   # numerical stability
        exp_w = np.exp(logits)
        return exp_w / exp_w.sum()

