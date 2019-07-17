from src.envs import wh_objects as wo

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

PATH_TO_CATALOG = '/Users/mmatskevichus/Desktop/reinforcement_learning/data/categories/yandex-market-leptops.csv'


def init_shelf(max_volume = 100, max_weight = 100, silent=True):
    shelf = wo.Shelf(
        max_volume=max_volume,
        max_weight=max_weight,
        silent=silent
    )
    return shelf


def init_wh_map(vis_map, path_to_catalog=PATH_TO_CATALOG, max_volume=200, max_weight=100,  silent=True):
    wo_map = []
    storage_worker = wo.StorageWorker(path_to_catalog)
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
                wo_unit = init_shelf(silent=silent, max_volume=max_volume, max_weight=max_weight)  # (coordinates=(i, j))
                response = storage_worker.check_shelf(max_weigth=wo_unit.available_load, max_size=wo_unit.free_volume)
                if response == 1:
                    for prod in storage_worker.prod_to_place:
                        wo_unit.put_product(prod)
                wo_row.append(wo_unit)
            elif sprite == '$':
                wo_unit = wo.PickPoint()  # (coordinates=(i, j))
                wo_row.append(wo_unit)
            else:
                print('Wrong sprite!')
                break
        wo_map.append(wo_row)
    return wo_map

if __name__ == '__main__':
    init_wh_map(vis_map=wh_vis_map, path_to_catalog=PATH_TO_CATALOG, max_volume=100, max_weight=200, silent = True)