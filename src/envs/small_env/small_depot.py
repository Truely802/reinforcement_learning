import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+---------+",
    "|P:P: :P:P|",
    "| : : : : |",
    "| : : : : |",
    "| | : : : |",
    "|D| : : : |",
    "+---------+",
]


class Depot(discrete.DiscreteEnv):


    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, num_rows = 5, num_columns =5):
        self.desc = np.asarray(MAP, dtype='c')

        self.prod_loc = prod_loc = [(4, 0), (0, 0), (0, 1), (0, 3), (0, 4)] #delivery point should be the first
        self.num_rows = num_rows
        self.num_columns = num_columns
        num_states = self.num_rows*self.num_columns*len(self.prod_loc) * len(self.prod_loc)
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}

        for row in range(num_rows):
            for col in range(num_columns):
                for prod_idx in range(1, len(prod_loc)):
                    dest_idx = 0
                    state = self.encode(row, col, prod_idx, dest_idx)
                    if prod_idx < len(prod_loc) and prod_idx != dest_idx:
                        initial_state_distrib[state] += 1
                    for action in range(num_actions):
                        # defaults
                        new_row, new_col, new_pass_idx = row, col, prod_idx
                        reward = -1  # default reward when there is no pickup/dropoff
                        done = False
                        worker_loc = (row, col)

                        if action == 0:
                            new_row = min(row + 1, max_row)
                        elif action == 1:
                            new_row = max(row - 1, 0)
                        if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                            new_col = min(col + 1, max_col)
                        elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                            new_col = max(col - 1, 0)
                        elif action == 4:  # pickup
                            if (prod_idx < len(self.prod_loc)-1 and worker_loc == prod_loc[prod_idx]):
                                new_pass_idx = 4
                            else:  # passenger not at location
                                reward = -10
                        elif action == 5:  # dropoff
                            if (worker_loc == prod_loc[dest_idx]) and prod_idx == 4:
                                new_pass_idx = dest_idx
                                done = True
                                reward = 20
                            elif (worker_loc in prod_loc) and prod_idx == 4:
                                new_pass_idx = prod_loc.index(worker_loc)
                            else:  # dropoff at wrong location
                                reward = -10
                        new_state = self.encode(
                            new_row, new_col, new_pass_idx, dest_idx)
                        P[state][action].append(
                            (1.0, new_state, reward, done))

        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def encode(self, worker_row, worker_col, prod_loc, dest_idx):
        i = worker_row
        i *= self.num_rows
        i += worker_col
        i *= self.num_columns
        i += prod_loc
        i *= len(self.prod_loc)
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % len(self.prod_loc))
        i = i // len(self.prod_loc)
        out.append(i % self.num_columns)
        i = i // self.num_columns
        out.append(i % self.num_rows)
        i = i // self.num_rows
        out.append(i)
        assert 0 <= i < max(self.num_rows, self.num_columns)
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        worker_row, worker_col, prod_idx, dest_idx = self.decode(self.s)
        print(worker_row, worker_col, prod_idx, dest_idx)
        def ul(x):
            return "_" if x == " " else x

        if prod_idx < len(self.prod_loc)-1:
            out[1 + worker_row][2 * worker_col + 1] = utils.colorize(
                out[1 + worker_row][2 * worker_col + 1], 'yellow', highlight=True)
            pi, pj = self.prod_loc[prod_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:
            out[1 + worker_row][2 * worker_col + 1] = utils.colorize(
                ul(out[1 + worker_row][2 * worker_col + 1]), 'green', highlight=True)

        di, dj = self.prod_loc[0]
        out[di+1][dj+1] = utils.colorize(out[di+1][2*dj+1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


