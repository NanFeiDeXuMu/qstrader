from qstrader.alpha_model.env_setup import QSTraderExecutionEnv
from qstrader import settings as qstrader_settings

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


class SyncVecNormalize(BaseCallback):
    """Copy running obs stats from train_env into eval_env before each evaluation.

    EvalCallback creates a separate VecNormalize wrapping eval_env with
    training=False, so its mean/var never update. Without syncing, the eval
    env normalises observations with stale (initial) statistics while the
    policy was trained on continuously-updated statistics — the mismatch makes
    checkpoint selection unreliable.
    """
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        return True


def _make_env(config):
    def _init():
        from qstrader import settings as qstrader_settings
        qstrader_settings.set_print_events(False)
        return Monitor(QSTraderExecutionEnv(config))
    return _init


def main():
    qstrader_settings.set_print_events(False)
    training_config = {
        'state_dim': 15,                                          # 5 assets x 3 features
        'action_dim': 5,
        'starting_day': '2010-01-01',
        'ending_day': '2018-12-31',
        'symbols': ['SPY', 'AGG', 'GLD', 'SHY', 'TLT'],
        'assets': ['EQ:SPY', 'EQ:AGG', 'EQ:GLD', 'EQ:SHY', 'EQ:TLT']
    }
    # Eval env uses a fixed 2019 window (not random sampling) for stable comparisons.
    # starting_day is set to 2018-11-01 to provide the 25-day burn-in period that
    # env_setup requires before any valid episode start can be sampled; the episode
    # itself will always land in early January 2019 (the only valid start range).
    eval_config = {
        **training_config,
        'starting_day': '2018-11-01',
        'ending_day': '2019-12-31'
    }

    N_ENVS = 4
    train_env = SubprocVecEnv([_make_env(training_config) for _ in range(N_ENVS)])
    # VecNormalize: normalise observations (running mean/std) and rewards (running std).
    # Critical for log-return rewards which are small (~1e-3) and for observation
    # features that span different scales across assets and market regimes.
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval env shares the same obs normalisation statistics as train_env (norm_reward
    # disabled for eval so EvalCallback sees true episode returns for comparison).
    eval_env = SubprocVecEnv([_make_env(eval_config)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)

    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,   # non-zero entropy bonus encourages exploration and prevents
                         # premature convergence to a uniform-weight local optimum
        verbose=1
    )

    os.makedirs('./ppo_checkpoints/', exist_ok=True)
    sync_cb = SyncVecNormalize(train_env, eval_env)
    callbacks = [
        CheckpointCallback(save_freq=10000, save_path='./ppo_checkpoints/', name_prefix='ppo_model'),
        EvalCallback(
            eval_env,
            eval_freq=50000,
            best_model_save_path='./ppo_checkpoints/best/',
            callback_before_eval=sync_cb,
        )
    ]

    model.learn(total_timesteps=500_000, callback=callbacks, progress_bar=True)
    model.save('ppo_final_model.zip')
    # Save VecNormalize statistics alongside the model so inference can reuse them.
    train_env.save('ppo_vecnormalize.pkl')


if __name__ == '__main__':
    main()
