import random
import numpy as np
from tqdm import tqdm
import os
from IPython.display import clear_output
from time import sleep
import pickle


class QTable(object):
    def __init__(self, environment, alpha=0.1, gamma=0.6, epsilon=0.2, verbose=False):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor, determines importance to future rewards
        self.epsilon = epsilon  # explore action space
        self.stats = dict()  # training stats
        self.env = environment
        self.encoder = dict()
        self.q_table = [[0] * self.env.action_space.n]
        self.frames = []
        self.verbose = verbose

    def _encode_state(self, state):
        state = tuple(state)
        if state not in self.encoder:
            self.encoder[state] = len(self.encoder)
        return self.encoder[state]

    def train(self, n_epoch):
        for i in tqdm(range(1, n_epoch + 1), position=0):
            state = self._encode_state(self.env.reset())
            self.stats['epochs'] = 0
            self.stats['penalties'] = 0
            self.stats['reward'] = 0
            done = False

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()  # Explore action space
                else:
                    action = np.argmax(self.q_table[state])  # Exploit learned values

                next_state, reward, done, info = self.env.step(action)
                if state < len(self.q_table):
                    self.q_table.append([0] * self.env.action_space.n)
                old_value = self.q_table[state][action]
                next_state = self._encode_state(next_state)
                next_max = np.max(self.q_table[next_state])

                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.q_table[state][action] = new_value

                if reward < -10:
                    self.stats['penalties'] += 1

                if reward > 0:
                    self.stats['reward'] += 1

                state = next_state
                self.stats['epochs'] += 1
        if self.verbose:
            print("Training finished.\n")

    def evaluate_performance(self, episodes=100):
        total_epochs, total_penalties = 0, 0

        for _ in range(episodes):
            state = self.env.reset()
            state = self._encode_state(state)
            epochs, penalties, rewards = 0, 0, 0

            done = False

            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, info = self.env.step(action)
                state = self._encode_state(state)
                if reward < -10:
                    penalties += 1
                if reward > 0:
                    rewards += 1

                epochs += 1

            total_penalties += penalties
            total_epochs += epochs
        if self.verbose:
            print(f"Results after {episodes} episodes:")
            print(f"Average timesteps per episode: {total_epochs / episodes}")
            print(f"Average penalties per episode: {total_penalties / episodes}")
            print(f"Average rewards per episode: {rewards / episodes}")

        return {
            'avg_timesteps': total_epochs / episodes,
            'avg_penalties': total_penalties / episodes,
            'avg_rewards': rewards / episodes
        }

    def operate(self):
        epochs, penalties, rewards = 0, 0, 0

        done = False

        state = self.env.reset()
        state = self._encode_state(state)
        while not done:
            action = np.argmax(self.q_table[state])
            state, reward, done, info = self.env.step(action)
            state = self._encode_state(state)
            if reward < -10:
                penalties += 1

            # Put each rendered frame into dict for animation
            self.frames.append({
                'frame': self.env.get_sprite_screen(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            })

            epochs += 1
        if self.verbose:
            print(f"Timesteps taken: {epochs}")
            print(f"Penalties incurred: {penalties}")
            print(f"Rewards incurred: {rewards}")

    def show_operation(self, sleep_time=.1):
        for i, frame in enumerate(self.frames):
            os.system('clear')
            clear_output(wait=True)
            print(frame['frame'])
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(sleep_time)

    def save_model(self, path):
        with open(path + '.pickle', 'wb') as f:
            pickle.dump(self.q_table, f)
            if self.verbose:
                print('Model saved.')

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)
            if self.verbose:
                print('Model loaded.')
