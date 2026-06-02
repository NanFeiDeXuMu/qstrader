from qstrader.alpha_model.env_setup import QSTraderExecutionEnv

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def main():
    training_config = {
        'state_dim': 15,                                          # 5 assets x 3 features
        'action_dim': 5,
        'starting_day': '2010-01-01',
        'ending_day': '2018-12-31',
        'symbols': ['SPY', 'AGG', 'GLD', 'IEI', 'TLT'],
        'assets': ['EQ:SPY', 'EQ:AGG', 'EQ:GLD', 'EQ:IEI', 'EQ:TLT']
    }
    eval_config = {
        **training_config,
        'starting_day': '2019-01-01',
        'ending_day': '2019-12-31'
    }

    train_env = DummyVecEnv([lambda: Monitor(QSTraderExecutionEnv(training_config))])
    eval_env = Monitor(QSTraderExecutionEnv(eval_config))

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
        ent_coef=0.0,
        verbose=1
    )

    os.makedirs('./ppo_checkpoints/', exist_ok=True)
    callbacks = [
        CheckpointCallback(save_freq=10000, save_path='./ppo_checkpoints/', name_prefix='ppo_model'),
        EvalCallback(eval_env, eval_freq=50000, best_model_save_path='./ppo_checkpoints/best/')
    ]

    model.learn(total_timesteps=500_000, callback=callbacks, progress_bar=True)
    model.save('ppo_final_model.zip')


if __name__ == '__main__':
    main()
