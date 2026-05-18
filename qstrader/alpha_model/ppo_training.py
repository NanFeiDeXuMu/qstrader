from env_setup import QSTraderExecutionEnv

'''
1. 准备数据
2. 定义环境
3. 基于PPO算法训练智能体
'''

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def main():
    training_config = {
        'target_quantity': 1000,
        'state_dim': 10,
        'action_dim': 5,
        'starting_day': '2010-01-01',
        'ending_day': '2018-12-31'
    }

    raw_env = QSTraderExecutionEnv(
        training_config=training_config,
        pcm=None,
        execution_handler=None,
        portfolio=None,
        slices=10
    )
    env = Monitor(raw_env)
    env = DummyVecEnv([lambda: env])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./ppo_checkpoints/',
        name_prefix='ppo_model'
    )

    model.learn(
        total_timesteps=100000,
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save("ppo_final_model.zip")

if __name__ == "__main__":
    main()