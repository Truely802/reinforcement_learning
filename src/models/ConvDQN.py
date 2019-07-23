import numpy as np
import os
import cv2


from collections import Counter, deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam


def dqn_model(input_shape, n_actions):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_actions))
    model.compile(Adam(), 'mse')
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
    return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]#current obs,  act, next_obs, reward, done


def train(out_dir, env, model, input_shape, n_actions,  num_episodes, buffer_len=20000, batch_size=48, discount_factor = 0.97):

    global_step = 0
    steps_train = 4
    start_steps = 2000
    exp_buffer = deque(maxlen=buffer_len)

    for e in range(num_episodes):
        epoch = 0
        env.reset()
        obs, reward, done, _ = env.step(1)
        episodic_reward = 0
        actions_counter = Counter()
        print('Episode %s'%e)
        os.makedirs(os.path.join(out_dir, 'Episode %s' %e))
        c = 0
        while not done:
            if c >  2000:
                break
            actions = model.predict(obs.reshape((1, ) + input_shape))   # feed the game screen and get the Q values for each action
            action = np.argmax(actions, axis=-1)[0]  # get the action
            actions_counter[str(action)] += 1
            action = epsilon_greedy(action, global_step, n_actions)   # select the action using epsilon greedy policy
            next_obs, reward, done, _ = env.step(action)   #  perform the action and move to the next state, next_obs, receive reward
            c+=1
            img_name = str(c) + '. ' + 'Action: ' + str(action) + ' ' + 'Reward: ' + str(reward) + ' ' + str(done)
            cv2.imwrite(os.path.join(out_dir, 'Episode %s' %e, img_name +'.png'),next_obs)
            exp_buffer.append([obs, action, next_obs, reward, done])
            if global_step % steps_train == 0 and global_step > start_steps:
                o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(exp_buffer, batch_size)
                for i in range(batch_size):
                    target = o_rew[i]
                    if not o_done[i]:
                        target = o_rew[i] + discount_factor * np.amax(model.predict(o_next_obs[i].reshape((1, ) + input_shape))[0])
                    target_f = model.predict(o_obs[i].reshape((1, ) + input_shape))
                    target_f[0][o_act[i]] = target
                    model.train_on_batch(o_obs[i].reshape((1, ) + input_shape), target_f)

            obs = next_obs
            epoch += 1
            global_step += 1

        # model.save(os.path.join(path_for_model,  epoch + '_' + 'cdqn.h5'))
        print('Epoch', epoch, 'Reward', episodic_reward)
    #TODO: 1) save model 2) predict script to check how it works
    return model