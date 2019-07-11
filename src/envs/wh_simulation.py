from src.envs import wh_map as wm
from src.envs import wh_objects as wo

import os


def render_map(map_obj, agent_obj):
    for i, row in enumerate(map_obj):
        to_print = list()
        for j, obj in enumerate(row):
            if (i, j) == agent_obj.coordinates:
                to_print.append(agent_obj.sprite)
            else:
                to_print.append(obj.sprite)
        print(''.join(to_print))


def sim_loop():
    map_obj = wm.init_wh_map(wm.wh_vis_map)
    agent_obj = wo.Agent(
        coordinates=(18, 9)
    )
    available_actions = {'w', 'a', 's', 'd', 'q', 't', 'g', 'i', 'r'}
    score = 0
    render_map(map_obj, agent_obj)
    while True:
        while True:
            action = input()
            if action in available_actions:
                break
            print('Unknown command. Try again.')
        os.system('clear')
        if action == 'w':
            r = agent_obj.move(to='u', map_obj=map_obj)
            if r == 0:
                score -= 10
        elif action == 'a':
            r = agent_obj.move(to='l', map_obj=map_obj)
            if r == 0:
                score -= 10
        elif action == 's':
            r = agent_obj.move(to='d', map_obj=map_obj)
            if r == 0:
                score -= 10
        elif action == 'd':
            r = agent_obj.move(to='r', map_obj=map_obj)
            if r == 0:
                score -= 10
        elif action == 'q':
            print('Breaking simulation.')
            break
        elif action == 't':
            r = agent_obj.take_product(product_name='MacBookPro', map_obj=map_obj)
            if r == 0:
                score -= 10
        elif action == 'g':
            r = agent_obj.put_product(product_name='MacBookPro', map_obj=map_obj)
            if r == 0:
                score -= 10
            elif r < 0:
                score += r
            elif r == 10:
                print('Customer satisfied!')
                score += 500
        elif action == 'i':
            r = agent_obj.inspect_shelf(map_obj=map_obj)
            if r == 0:
                score -= 10
            else:
                print(r)
        elif action == 'r':
            print('Waiting...')
        score -= 10
        render_map(map_obj, agent_obj)
        print(f'Score: {score}')


if __name__ == '__main__':
    sim_loop()
