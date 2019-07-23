import os
from src.envs.wh_env import WarehouseEnv
from src.models.ConvDQN import train, dqn_model

env = WarehouseEnv(frequency=-1)
env.reset()
env.step(1)
q_model = dqn_model(env.wh_shape, n_actions=len(env.actions))
num_episodes = 100000
path_for_model = './data/saved_model'
model = train(
      out_dir= './data/screens',
      env=env,
      model=q_model,
      input_shape=env.wh_shape,
      n_actions=len(env.actions),
      num_episodes=num_episodes,
      buffer_len=20000,
      batch_size=48,
      discount_factor=0.97)
