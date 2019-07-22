from src.envs.wh_env import WarehouseEnv
from src.models.ConvDQN import train, dqn_model
import cv2 as cv

if __name__ == '__main__':
    input_shape = (23, 20, 1)
    env = WarehouseEnv()
    q_model = dqn_model(input_shape, n_actions=env.n_actions)
    num_episodes = 100000
    train(env=env,
          model=q_model,
          input_shape=input_shape,
          n_actions=env.n_actions,
          num_episodes=num_episodes,
          buffer_len=20000,
          batch_size=48,
          discount_factor=0.97)