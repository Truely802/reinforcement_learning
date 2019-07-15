from src.envs import wh_objects as wo
import numpy as np

wh_vis_map = [
    '++++++++++++++++++++',
    '+.#..#..#..#..#..#.+',
    '+.#..#..#..#..#..#.+',
    '+.#..#..#..#..#..#.+',
    '+.#..#..#..#..#..#.+',
    '+..................+',
    '+..................+',
    '+.#######..#######.+',
    '+..................+',
    '+..................+',
    '+.#######..#######.+',
    '+..................+',
    '+..................+',
    '+.##..##..##..##..#+',
    '+.##..##..##..##..#+',
    '+.##..##..##..##..#+',
    '+.##..##..##..##..#+',
    '+..................+',
    '+..................+',
    '+..................+',
    '+$$$$$$$$$$$$$$$$$$+'
]


def init_shelf(random=True, silent=True):
    shelf = wo.Shelf(
        # coordinates=coordinates,
        max_volume=100,
        max_weight=100,
        silent=silent
    )
    if random:
        num = np.random.randint(1, 6)
    else:
        num = 1
    for _ in range(num):
        laptop = wo.Product(
            name='MacBookPro',
            weight=2.77,
            volume=10.989,
            manufacturer='Apple',
            price=2000
        )
        shelf.put_product(laptop)
    return shelf


def init_wh_map(vis_map, random=True, silent=True):
    wo_map = []
    for i, row in enumerate(vis_map):
        wo_row = []
        for j, sprite in enumerate(row):
            if sprite == '+':
                wo_unit = wo.Wall()  # (coordinates=(i, j))
                wo_row.append(wo_unit)
            elif sprite == '.':
                wo_unit = wo.SimpleFloor()  # (coordinates=(i, j))
                wo_row.append(wo_unit)
            elif sprite == '#':
                wo_unit = init_shelf(silent=silent, random=random)  # (coordinates=(i, j))
                wo_row.append(wo_unit)
            elif sprite == '$':
                wo_unit = wo.PickPoint()  # (coordinates=(i, j))
                wo_row.append(wo_unit)
            else:
                print('Wrong sprite!')
                break
        wo_map.append(wo_row)
    return wo_map
