from src.envs import wh_objects as wo
from src.utils import config as co

wh_vis_map = [
    '++++++++++++++++++++',
    # '+.#..#..#..#..#..#.+',
    # '+.#..#..#..#..#..#.+',
    # '+.#..#..#..#..#..#.+',
    # '+.#..#..#..#..#..#.+',
    # '+..................+',
    # '+..................+',
    # '+.#######..#######.+',
    # '+..................+',
    # '+..................+',
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

wh_map_medium = [
    '++++++++++++++++++++',
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

wh_map_small = [
    '++++++++++++++',
    '+#####..#####+',
    '+............+',
    '+............+',
    '+#####..#####+',
    '+............+',
    '+............+',
    '+$$$$$$$$$$$$+'
]


def init_shelf(max_volume=100, max_weight=100, silent=True):
    shelf = wo.Shelf(
        max_volume=max_volume,
        max_weight=max_weight,
        silent=silent
    )
    return shelf


def init_wh_map(vis_map, df_catalog, max_volume=200, max_weight=100,  silent=True):
    wo_map = []
    storage_worker = wo.StorageWorker(df_catalog=df_catalog)
    for i, row in enumerate(vis_map):
        wo_row = []
        for j, sprite in enumerate(row):
            if sprite == '+':
                wo_unit = wo.Wall()
                wo_row.append(wo_unit)
            elif sprite == '.':
                wo_unit = wo.SimpleFloor()
                wo_row.append(wo_unit)
            elif sprite == '#':
                wo_unit = init_shelf(silent=silent, max_volume=max_volume, max_weight=max_weight)
                response = storage_worker.check_shelf(max_weigth=wo_unit.available_load, max_size=wo_unit.free_volume,
                                                      coordinates=(i, j))
                if response == 1:
                    for prod in storage_worker.prod_to_place:
                        wo_unit.put_product(prod)
                wo_row.append(wo_unit)
            elif sprite == '$':
                wo_unit = wo.PickPoint()
                wo_row.append(wo_unit)
            else:
                print('Wrong sprite!')
                break
        wo_map.append(wo_row)
    return wo_map, storage_worker.product_scheme


if __name__ == '__main__':
    import pandas as pd
    df_catalog = pd.read_csv(co.PATH_TO_CATALOG, index_col=0).fillna(0)
    wo_map, prod_scheme = init_wh_map(
        vis_map=wh_vis_map,
        df_catalog=df_catalog,
        max_volume=100,
        max_weight=200,
        silent=True
    )

    print(prod_scheme)