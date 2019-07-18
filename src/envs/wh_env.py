import gym
import numpy as np
from gym import spaces

from src.envs import wh_map as wm
from src.envs import wh_objects as wo
from src.utils import config as co
import PIL

import os


class WarehouseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'ansi']
    }

    def __init__(self, map_sketch=wm.wh_vis_map, num_turns=None, max_order_line=25, frequency=0.05,
                 simplified_state=False, silent=True):
        self.map_sketch = map_sketch
        self.silent = silent
        self.frequency = frequency
        self.load_map = wm.init_wh_map
        self.simplified_state = simplified_state
        self.map, self.product_scheme = self.load_map(
            self.map_sketch,
            max_weight=200,
            max_volume=100,
            path_to_catalog=co.PATH_TO_CATALOG,
            silent=self.silent
        )
        self.agent = wo.Agent(
            coordinates=(18, 9),
            silent=self.silent,
            max_weight=200,
            max_volume=100,
            frequency=self.frequency,
            product_scheme=self.product_scheme
        )

        self.actions = {
            'w': lambda x, y: x.move(to='u', map_obj=y),      # 0
            'a': lambda x, y: x.move(to='l', map_obj=y),      # 1
            's': lambda x, y: x.move(to='d', map_obj=y),      # 2
            'd': lambda x, y: x.move(to='r', map_obj=y),      # 3
            't': lambda x, y: x.take_product(map_obj=y),      # 4
            'g': lambda x, y: x.deliver_products(map_obj=y),  # 5
            'i': lambda x, y: x.inspect_shelf(map_obj=y),     # 6
            'r': lambda x, _: x.wait(),                       # 7
            # 'q': 'break_loop',                              # N/A
        }

        self.action_space = spaces.Discrete(len(self.actions))
        if self.simplified_state:
            self.observation_space = spaces.Box(
                low=np.array([1, 1, 0, 0]),
                high=np.array([
                    len(self.map[0])-2,
                    len(self.map)-2,
                    self.agent.max_weight,
                    self.agent.max_volume
                ]),
                dtype=np.uint32
            )
        else:
            self.observation_space = None  # TODO: finish observation space for product scheme (?)

        self.reward_policy = {
            2: 50,
            1: -1,
            0: -10,
            10: 500,  # done
            -1: -1000  # drop
        }

        if num_turns is None:
            self.num_turns = np.inf
        else:
            self.num_turns = num_turns

        if max_order_line is None:
            self.max_order_line = np.inf
        else:
            self.max_order_line = max_order_line

        self.turns_left = self.num_turns
        self.score = 0
        self.encode = self._get_action_code()
        self.prod_coord = None

    def create_wh_screen(self, wh_vis_map):
        wh_vis_map = wh_vis_map.split('\n')
        length = len(wh_vis_map[0])
        width = len(wh_vis_map)
        screen = np.zeros((width, length))

        for i, row in enumerate(wh_vis_map):
            for j, sprite in enumerate(row):
                if sprite == '.':
                    screen[i, j] = 0.
                if sprite == '#':
                    screen[i, j] = 0.2
                elif sprite == '$':
                    screen[i, j] = 0.4
                elif sprite == 'X':
                    screen[i, j] = 0.6
                elif sprite == 'P':
                    screen[i, j] = 0.9
                else:
                    screen[i, j] = 1.

        # remove borders
        screen = screen[1:, 1:-1]
        return PIL.Image.fromarray(screen*255)

    def _get_action_code(self):
        acts = dict()
        for i, act in enumerate(self.actions.keys()):
            acts[i] = act

        return acts

    def step(self, action_code):
        self.agent.order_list()
        action = self.encode[action_code]
        response = self.actions[action](self.agent, self.map)

        if not isinstance(response, int):
            reward = 0
            if not self.silent: print(response)
        elif response in self.reward_policy:
            reward = self.reward_policy[response]
        elif response > 0 and response % 10 == 0:
            reward = 0
            for _ in range(response // 10):
                reward += self.reward_policy[10]
        else:
            reward = responce
        self.score += reward
        self.turns_left -= 1

        if reward == 500:
            done=True
        else:
            done=False

        # observation = []  # self.map2feats(self.map, self.agent)
        # observation += [*self.agent.coordinates, self.agent.free_volume, self.agent.available_load]
        # if self.turns_left <= 0 or len(self.agent.order_list) > self.max_order_line:
        #     done = True
        # else:
        #     done = False

        info = self.agent.order_list.__str__()
        screen = self.render()
        
        return self.create_wh_screen(screen), reward, done, {'action': action_code, 'order_list': info}

    def reset(self):
        self.map, self.product_scheme = self.load_map(
            self.map_sketch,
            max_weight=200,
            max_volume=100,
            path_to_catalog=co.PATH_TO_CATALOG,
            silent=self.silent
        )
        self.agent = wo.Agent(
            coordinates=(18, 9),
            silent=self.silent,
            max_weight=200,
            max_volume=100,
            frequency=self.frequency,
            product_scheme=self.product_scheme
        )
        
        screen = self.render()
        return self.create_wh_screen(screen)

    def render(self, mode='human'):  # TODO: finish respriting for shelves with products on product list
        picture = []
        for i, row in enumerate(self.map):
            to_print = list()
            for j, obj in enumerate(row):
                if (i, j) == self.agent.coordinates:
                    to_print.append(self.agent.sprite)

                    elif any([(i, j) in self.agent.order_list.product_scheme[prod]
                          for prod in self.agent.order_list.order_list.keys()]):
                    to_print.append('P')
                else:
                    to_print.append(obj.sprite)
            picture.append(''.join(to_print))
        return '\n'.join(picture)

    def close(self):
        os.system('clear')
        
        