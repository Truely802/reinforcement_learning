from IPython.display import clear_output
from time import sleep
import numpy as np
import random

from src.envs.small_env.small_depot import Depot

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

if __name__ == '__main__':

    exp_n = 10
    res = []
    for exp in range(exp_n):

        num_cols = 5
        num_rows = 5

        env = Depot(num_cols, num_rows)
        q_table = np.zeros([env.observation_space.n, env.action_space.n])

        print("""Training the agent""")

        # Hyperparameters
        alpha = 0.1 #learning rate
        gamma = 0.5 #discount factor
        epsilon = 0.2 #tradeoff

        # For plotting metrics
        all_epochs = []
        all_penalties = []

        for i in range(1, 10001):
            state = env.reset()

            epochs, penalties, reward, = 0, 0, 0
            done = False

            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()  # Explore action space
                else:
                    action = np.argmax(q_table[state])  # Exploit learned values

                next_state, reward, done, info = env.step(action)
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1

            if i % 100 == 0:
                clear_output(wait=True)
                #print(f"Episode: {i}")

        print("Training finished.\n")

        """Evaluate agent's performance after Q-learning"""

        total_epochs, total_penalties = 0, 0
        episodes = 100
        frames = []
        done = False


        for _ in range(episodes):
            state = env.reset()
            epochs, penalties, reward = 0, 0, 0

            while not done:
                action = np.argmax(q_table[state])
                state, reward, done, info = env.step(action)

                if reward == -10:
                    penalties += 1

                frames.append({
                    'frame': env.render(mode='ansi'),
                    'state': state,
                    'action': action,
                    'reward': reward
                }
                )
                epochs += 1

            total_penalties += penalties
            total_epochs += epochs
        res.append(total_epochs)
        print_frames(frames)
        print(f"Stop after {total_epochs} episodes:")

    print(np.mean(res), 'avg. number of steps')
    print(res)
