import numpy as np

from collections import Counter, deque
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten
from keras.optimizers import RMSprop


def create_wh_sreen(wh_vis_map):
    length = len(wh_vis_map[0])
    width = len(wh_vis_map)
    X = np.zeros((width, length))

    for i, row in enumerate(wh_vis_map):
        for j, sprite in enumerate(row):
            if sprite == '.':
                X[i, j]=0.
            if sprite == '#':
                X[i, j]=1.
            elif sprite == '$':
                X[i, j]=2.
            elif sprite == 'X':
                X[i, j]=3.
            else:
                X[i, j]=0.

    #remove borders
    X = X[1:, 1:-1]
    return X


def dqn_model(input_shape, n_actions):
    model = Sequential()
    model.add(Convolution2D(16, nb_row=3, nb_col=3, input_shape=input_shape, activation='relu'))
    model.add(Convolution2D(16, nb_row=3, nb_col=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_actions))
    model.compile(RMSprop(), 'MSE')
    model.summary()

    return model

def epsilon_greedy(action, step, n_actions):

    eps_min = 0.05
    eps_max = 1.0
    eps_decay_steps = 500000

    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return action

def sample_memories(exp_buffer, batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]  #current obs,  act, next_obs, reward, done

def train(env, model, num_episodes, buffer_len=20000, batch_size=48, discount_factor = 0.97):

    global_step = 0
    steps_train = 4
    start_steps = 2000
    exp_buffer = deque(maxlen=buffer_len)

    for i in range(num_episodes):
        env.reset()
        screen = env.render()
        done = False
        obs = create_wh_sreen(screen)
        epoch = 0
        episodic_reward = 0
        actions_counter = Counter()
        episodic_loss = []

        # while the state is not the terminal state
        while not done:

            actions = model.predict(obs)   # feed the game screen and get the Q values for each action
            action = np.argmax(actions, axis=-1)   # get the action
            actions_counter[str(action)] += 1
            action = epsilon_greedy(action, global_step)   # select the action using epsilon greedy policy
            next_obs, reward, done= env.step(action)   # now perform the action and move to the next state, next_obs, receive reward

            exp_buffer.append([obs, action, next_obs, reward, done])

            # After certain steps, we train our Q network with samples from the experience replay buffer
            if global_step % steps_train == 0 and global_step > start_steps:
                o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)
                o_obs = [x for x in o_obs]
                o_next_obs = [x for x in o_next_obs]

                next_act = model.predict(o_next_obs)
                y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1 - o_done)

                train_loss = model.train_on_batch(np.array(o_obs), np.array(y_batch))

                episodic_loss.append(train_loss)

            obs = next_obs
            epoch += 1
            global_step += 1
            episodic_reward += reward

        print('Epoch', epoch, 'Reward', episodic_reward)
    #TODO: 1) save model 2) predict script to check how it works
    return model