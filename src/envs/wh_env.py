import gym
import numpy as np
import pandas as pd
from gym import spaces

from src.envs import wh_map as wm
from src.envs import wh_objects as wo
import src.utils.config as co
from IPython.display import clear_output
from matplotlib import pyplot as plt


class WarehouseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'ipynb']
    }

    def __init__(self, map_sketch=wm.wh_vis_map, catalog=None, num_turns=None, max_order_line=25,
                 agent_max_load=200, agent_max_volume=1000, agent_start_pos=(18, 9),
                 shelf_max_load=200, shelf_max_volume=100,
                 frequency: float = 0.05, simplified_state: bool = False,
                 only_one_product: bool = False, win_size=(300, 300), silent: bool = True):
        self.map_sketch = map_sketch
        self.only_one_prod = only_one_product
        self.agent_max_load = agent_max_load
        self.agent_max_volume = agent_max_volume
        self.agent_start_pos = agent_start_pos
        self.shelf_max_weight = shelf_max_load
        self.shelf_max_volume = shelf_max_volume
        self.win_size = win_size
        self.silent = silent

        if catalog is None:
            self.df_catalog = pd.read_csv(co.PATH_TO_CATALOG, index_col=0).fillna(0)
        else:
            self.df_catalog = catalog

        if only_one_product:
            self.frequency = -1
        else:
            self.frequency = frequency

        self.load_map = wm.init_wh_map
        self.simplified_state = simplified_state
        self.viewer = None
        self.map, self.product_scheme = self.load_map(
            self.map_sketch,
            max_weight=self.shelf_max_weight,
            max_volume=self.shelf_max_volume,
            df_catalog=self.df_catalog,
            silent=self.silent
        )
        self.agent = wo.Agent(
            coordinates=self.agent_start_pos,
            silent=self.silent,
            max_weight=self.agent_max_load,
            max_volume=self.agent_max_volume,
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
            # 'i': lambda x, y: x.inspect_shelf(map_obj=y),   # N/A
            # 'r': lambda x, _: x.wait(),                     # N/A
            # 'q': 'break_loop',                              # N/A
        }

        self.action_space = spaces.Discrete(len(self.actions))
        if self.simplified_state:
            self.observation_space = spaces.Box(
                low=np.array([1, 1, 0, 0]),
                high=np.array([
                    len(self.map[0]) - 2,
                    len(self.map) - 2,
                    self.agent.max_weight,
                    self.agent.max_volume
                ]),
                dtype=np.uint32
            )
        else:
            self.observation_space = None

        self.reward_policy = {
            2: 50,
            # 1: -1,
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
        self.wh_shape = self.create_wh_screen(self.get_sprite_screen()).shape

    def return_manhattan_dist(self, if_prod):
        if  if_prod:
            dist = abs(self.agent.coordinates[0] - len(self.map_sketch[0]))
            return dist
        else:
            prod_dist = {}
            prod_coord = [self.agent.order_list.product_scheme[p] for p in self.agent.order_list.order_list.keys()][0]
            for c in prod_coord:
                prod_dist[c] = abs(self.agent.coordinates[0] - c[0]) + abs(self.agent.coordinates[1] - c[1])
            return prod_dist

    def calculate_reward(self, dist):
        rewards = [1 - (dist[prod_c] / self.max_dist[prod_c]) ** 0.4 for prod_c in self.max_dist.keys()]
        return min(rewards)

    def reward_func(self, response):
        if not isinstance(response, int):
            reward = 0
            if not self.silent:
                print(response)
        elif response in self.reward_policy:
            reward = self.reward_policy[response]
            if reward == 50:
                self.pickup_point = self.agent.coordinates

        elif response > 0 and response % 10 == 0:
            reward = 0
            for _ in range(response // 10):
                reward += self.reward_policy[10]

        elif response == 1:

            if self.agent.inventory:
                current_dist = self.return_manhattan_dist(if_prod= True)
                reward = 1 - current_dist / abs(self.pickup_point[0] - len(self.map_sketch[0]))

            else:
                current_dist = self.return_manhattan_dist(if_prod=False)
                reward = self.calculate_reward(current_dist)
        else:
            reward = response
        return reward

    def create_wh_screen(self, wh_vis_map):
        wh_vis_map = wh_vis_map.split('\n')
        length = len(wh_vis_map[0])
        width = len(wh_vis_map)
        screen = np.zeros((width, length))

        for i, row in enumerate(wh_vis_map):
            for j, sprite in enumerate(row):
                if sprite == '.':
                    screen[i, j] = 0.
                elif sprite == ' ':
                    screen[i, j] = 0.05
                elif sprite == '#':
                    screen[i, j] = 0.2
                elif sprite == '$':
                    screen[i, j] = 0.4
                elif sprite == 'X':
                    screen[i, j] = 0.6
                elif sprite == 'P':
                    screen[i, j] = 0.9
                elif sprite == '-':
                    screen[i, j] = 0.3
                elif sprite == '=':
                    screen[i, j] = 0.35
                else:
                    screen[i, j] = 1.
        screen = np.array(screen*255, dtype=np.uint8)[:, :, np.newaxis]
        return screen

    def _get_action_code(self):
        acts = dict()
        for i, act in enumerate(self.actions.keys()):
            acts[i] = act

        return acts

    def step(self, action_code):
        self.agent.order_list()
        self.max_dist = self.return_manhattan_dist(if_prod=False)
        action = self.encode[action_code]
        response = self.actions[action](self.agent, self.map)
        reward = self.reward_func(response)

        self.score += reward
        self.turns_left -= 1
        if self.simplified_state:
            observation = [*self.agent.coordinates, self.agent.free_volume, self.agent.available_load]
        else:
            screen = self.get_sprite_screen()
            observation = self.create_wh_screen(screen)

        if self.turns_left <= 0 or len(self.agent.order_list) > self.max_order_line or \
                        len(self.agent.order_list.list_of_products) == 0:
            done = True
        else:
            done = False

        if self.only_one_prod and reward == 500:
            done = True

        info = self.agent.order_list.__str__()

        return observation, reward, done, {'action': action_code, 'order_list': info}

    def reset(self):
        self.map, self.product_scheme = self.load_map(
            self.map_sketch,
            max_weight=self.shelf_max_weight,
            max_volume=self.shelf_max_volume,
            df_catalog=self.df_catalog,
            silent=self.silent
        )
        self.agent = wo.Agent(
            coordinates=self.agent_start_pos,
            silent=self.silent,
            max_weight=self.agent_max_load,
            max_volume=self.agent_max_volume,
            frequency=self.frequency,
            product_scheme=self.product_scheme
        )

        self.score = 0
        self.turns_left = self.num_turns

        if self.simplified_state:
            observation = [*self.agent.coordinates, self.agent.free_volume, self.agent.available_load]
        else:
            screen = self.get_sprite_screen()
            observation = self.create_wh_screen(screen)

        return observation

    def _get_status_bar(self, current, maximum, length, sprite, free_space_sprite=' '):
        bar = [free_space_sprite] * length
        num_to_fill = int(length * current/maximum)
        for i in range(num_to_fill):
            bar[i] = sprite
        return "".join(bar)

    def get_sprite_screen(self,):
        picture = []
        for i, row in enumerate(self.map):
            to_print = list()
            for j, obj in enumerate(row):
                if (i, j) == self.agent.coordinates:
                    to_print.append(self.agent.sprite)
                elif any([(i, j) in self.agent.order_list.product_scheme[prod]
                          for prod in self.agent.order_list.order_list.keys()
                          if prod not in self.agent.inventory.keys()]):
                    to_print.append('P')
                else:
                    to_print.append(obj.sprite)
            picture.append(''.join(to_print))
        picture.append(self._get_status_bar(
            current=self.agent.available_load,
            maximum=self.agent.max_weight,
            length=len(picture[0]),
            sprite='-',
            free_space_sprite=' '
        ))
        picture.append(self._get_status_bar(
            current=self.agent.free_volume,
            maximum=self.agent.max_volume,
            length=len(picture[0]),
            sprite='=',
            free_space_sprite=' '
        ))
        return '\n'.join(picture)

    def render(self, mode='human'):
        img = self.create_wh_screen(self.get_sprite_screen())
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            import cv2
            from gym.envs.classic_control import rendering
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, dsize=self.win_size, interpolation=cv2.INTER_NEAREST)
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == 'ipynb':
            clear_output(wait=True)
            plt.imshow(img.reshape(img.shape[0], img.shape[1]))
            plt.show()

    def close(self):
        clear_output(wait=True)
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
