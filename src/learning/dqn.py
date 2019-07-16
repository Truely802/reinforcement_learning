import random
import gym
import numpy as np


from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam

from src.envs.small_env.small_depot import Depot

EPISODES = 1000


if __name__ == "__main__":
    num_cols = 5
    num_rows = 5

    env = Depot(num_cols, num_rows)
    env.render()

    action_size = env.action_space.n
    state_size = env.observation_space.n

    print("Number of actions: %d" % env.action_space.n)
    print("Number of states: %d" % env.observation_space.n)

    np.random.seed(123)
    env.seed(123)

    model_only_embedding = Sequential()
    model_only_embedding.add(Embedding(state_size, action_size, input_length=1))
    model_only_embedding.add(Reshape((action_size,)))
    print(model_only_embedding.summary())

    model = Sequential()
    model.add(Embedding(state_size, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    print(model.summary())


    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy()
    dqn_only_embedding = DQNAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup=state_size,
                                  target_model_update=1e-2, policy=policy)

    dqn_only_embedding.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn_only_embedding.fit(env, nb_steps=100000, visualize=False, verbose=1, nb_max_episode_steps=99,
                           log_interval=10000)
    dqn_only_embedding.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=99)
