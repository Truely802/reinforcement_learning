import pandas as pd
import numpy as np


class Product(object):

    def __init__(self, name, weight, volume, manufacturer, price, n_purchase, category):
        self.name = name
        self.weight = weight
        self.volume = volume
        self.manufacturer = manufacturer
        self.price = price
        self.n_purchase = n_purchase  # for month
        self.category = category


class Shelf(object):

    def __init__(self, max_volume, max_weight, silent=True):
        self.max_volume = max_volume  # maximum volume in liters per 1 section
        self.max_weight = max_weight  # maximum weight in kilos
        self.products = dict()
        self.free_volume = max_volume
        self.available_load = max_weight
        self.origin = "shelf"
        self.sprite = "#"
        self.passable = False
        self.silent = silent  # silent mode

    def put_product(self, product):
        if self.free_volume < product.volume and self.available_load < product.weight:
            if not self.silent: print('The shelf is overburden.')
            return 0
        self.products[product.name] = self.products.get(product.name, {'product': product,  'count': 0})
        self.free_volume -= product.volume
        self.available_load -= product.weight
        self.products[product.name]['count'] += 1
        return 1

    def remove_product(self, product_name):
        if product_name not in self.products or self.products[product_name]['count'] == 0:
            if not self.silent: print('No such a product.')
            return 0
        self.free_volume += self.products[product_name]['product'].volume
        self.available_load += self.products[product_name]['product'].weight
        self.products[product_name]['count'] -= 1
        return self.products[product_name]['product']

    def inspect(self):
        if len(self.products) == 0:
            if not self.silent: print('The shelf is empty.')
            return 0
        return sorted([(k, v['count']) for (k, v) in self.products.items()], reverse=True, key=lambda x: x[1])


class Agent(object):

    def __init__(self, product_scheme: dict, name: str = 'Bill', max_volume: float = 30,
                 max_weight: float = 30, coordinates: tuple = (1, 1),
                 frequency: float = 0.2, silent: bool = True):
        self.name = name
        self.coordinates = coordinates
        self.max_volume = max_volume
        self.max_weight = max_weight
        self.free_volume = max_volume
        self.available_load = max_weight
        self.inventory = dict()
        self.origin = "agent"
        self.sprite = "X"
        self.passable = False
        self.silent = silent  # silent mode
        self.order_list = OrderList(product_scheme=product_scheme, frequency=frequency)

    def move(self, map_obj, to='u'):
        if to == 'u':
            dest = (self.coordinates[0] - 1, self.coordinates[1])
        elif to == 'd':
            dest = (self.coordinates[0] + 1, self.coordinates[1])
        elif to == 'l':
            dest = (self.coordinates[0], self.coordinates[1] - 1)
        elif to == 'r':
            dest = (self.coordinates[0], self.coordinates[1] + 1)
        else:
            if not self.silent: print('Wrong destination code.')
            return 0

        if not map_obj[dest[0]][dest[1]].passable:
            if not self.silent: print("Can't move here.")
            return 0
        self.coordinates = dest
        return 1

    def wait(self):
        return 1

    def _find_shelf(self, map_obj):
        neighbours_coordinates = [
            (self.coordinates[0] + 1, self.coordinates[1]),
            (self.coordinates[0] - 1, self.coordinates[1]),
            (self.coordinates[0], self.coordinates[1] + 1),
            (self.coordinates[0], self.coordinates[1] - 1),
        ]

        for (i, j) in neighbours_coordinates:
            if map_obj[i][j].origin == 'shelf' or map_obj[i][j].origin == 'pickpoint':
                return map_obj[i][j]
        return 0

    def put_product(self, product_name, map_obj):
        shelf = self._find_shelf(map_obj)
        if product_name not in self.inventory or self.inventory[product_name]['count'] == 0:
            if not self.silent: print('No such a product.')
            return 0
        if shelf == 0:
            if not self.silent: print('You\'ve broken', self.inventory[product_name]['product'].name)
            self.free_volume -= self.inventory[product_name]['product'].volume
            self.available_load -= self.inventory[product_name]['product'].weight
            self.inventory[product_name]['count'] -= 1
            return -1 * self.inventory[product_name]['product'].price
        self.free_volume -= self.inventory[product_name]['product'].volume
        self.available_load -= self.inventory[product_name]['product'].weight
        self.inventory[product_name]['count'] -= 1
        product = self.inventory[product_name]['product']
        response = shelf.put_product(product)
        return response

    def take_product(self, map_obj):
        shelf = self._find_shelf(map_obj)
        if shelf == 0:
            if not self.silent: print('No shelf here')
            return 0
        response = 0
        if len(self.order_list) == 0:
            return response
        for product_name, count in self.order_list:
            for _ in range(count):
                product = shelf.remove_product(product_name)
                if product == 0:
                    continue
                if self.free_volume < product.volume or self.available_load < product.weight:
                    if not self.silent: print('No more space in inventory')
                    self.put_product(product_name=product.name, map_obj=map_obj)
                    continue
                self.inventory[product.name] = self.inventory.get(product.name, {'product': product, 'count': 0})
                self.free_volume -= product.volume
                self.available_load -= product.weight
                self.inventory[product.name]['count'] += 1
                response = 2
        return response

    def deliver_products(self, map_obj):
        count = 0
        for product_name in self.inventory:
            for _ in range(np.minimum(  # Sometimes agent tries to give more products, than it could be given.
                    self.inventory[product_name]['count'],
                    self.order_list.get(product_name, 0)
            )):
                response = self.put_product(product_name, map_obj)
                if response <= 0:
                    return response
                count += 1
                self.order_list.pop(product_name)
        return 10 * count

    def inspect_shelf(self, map_obj):
        shelf = self._find_shelf(map_obj)
        if not isinstance(shelf, int):
            return shelf.inspect()
        return 0


class PickPoint(Shelf):

    def __init__(self):
        super().__init__(max_volume=None, max_weight=None)
        self.origin = "pickpoint"
        self.sprite = "$"
        self.passable = False

    def put_product(self, product):
        return 10

    def remove_product(self, product_name):
        return 0

    def inspect(self):
        return 0


class SimpleFloor(object):

    def __init__(self):
        self.passable = True
        self.origin = "floor"
        self.sprite = "."


class Wall(object):

    def __init__(self):
        self.passable = False
        self.origin = "wall"
        self.sprite = "+"


class StorageWorker(object):

    def __init__(self, path_to_catalog, single=False):
        self.catalog = pd.read_csv(path_to_catalog, index_col=0).fillna(0)
        self.list_of_products = self._gather_products()
        self.prod_to_place = list()
        self.prod_on_shelfs = list()
        self.product_scheme = dict()
        self._weighted_list, self._max_weight = self._get_weighted_list()
        self.single = single

    def _get_weighted_list(self):
        sum_of_events = np.sum([prod.n_purchase for prod in self.list_of_products])
        weighted_list = dict()
        counter = 0
        for product in self.list_of_products:
            odd = product.n_purchase / sum_of_events
            if odd * 1e5 < 1:
                weight = 1
            else:
                weight = int(odd * 1e5)
            counter += weight
            weighted_list[counter] = product
        return weighted_list, counter

    def _sample_product(self):
        sample_seed = np.random.randint(0, self._max_weight)
        ordered_weights = sorted(self._weighted_list.keys(), reverse=False)
        for i, weight in enumerate(ordered_weights):
            if weight <= sample_seed < ordered_weights[i+1]:
                return self._weighted_list[weight]

    def _gather_products(self):
        list_of_products = []
        for product_n in range(self.catalog.shape[0]):
            list_of_products.append(Product(
                name=self.catalog.iloc[product_n]['name'] ,
                weight=self.catalog.iloc[product_n]['weigth'],
                volume=self.catalog.iloc[product_n]['volume'],
                manufacturer=None,
                price=self.catalog.iloc[product_n]['price'],
                n_purchase=self.catalog.iloc[product_n]['purchase'],
                category='laptop'
            ))
        return list_of_products

    def check_shelf(self, max_weigth, max_size, coordinates):
        prod_to_place = list()
        total_weight, total_volume = 0, 0
        if len(self.list_of_products) != 0:
            # for prod in self.list_of_products:
            while True:
                if self.single:  # TODO: Finish 1-prod init
                    cnt = 0
                    while True:
                        prod = self._sample_product()
                        cnt += 1
                        if prod not in self.prod_on_shelfs or cnt > 1e7:
                            break
                else:
                    prod = self._sample_product()
                    if not prod:  # Sometimes samples None's. Don't know why.
                        continue
                if total_weight <= max_weigth and total_volume <= max_size:
                    prod_to_place.append(prod)
                    total_weight += prod.weight
                    total_volume += prod.volume
                    coord_list = self.product_scheme.get(prod.name, list())
                    coord_list.append(coordinates)
                    self.product_scheme[prod.name] = coord_list
                else:
                    break
            self.prod_to_place = prod_to_place
            self.prod_on_shelfs = self.prod_on_shelfs + prod_to_place
            # self.list_of_products = [x for x in self.list_of_products if x not in self.prod_on_shelfs]
            return 1
        else:
            return 0


class OrderList(object):

    def __init__(self, product_scheme: dict, frequency: float = 0.2):
        self.product_scheme = product_scheme
        self.list_of_products = self._gather_products()
        # self._weighted_list, self._max_weight = self._get_weighted_list()
        self.frequency = frequency
        self.order_list = dict()

    def __len__(self):
        return int(np.sum([v for v in self.order_list.values()]))

    def __getitem__(self, item):
        return self.order_list[item]

    def __iter__(self):
        return iter(self.order_list.items())

    def __contains__(self, item):
        return item in self.order_list

    def __call__(self):
        seed = np.random.uniform()
        if seed < self.frequency:
            sample_product = self._sample_product()
            num = self.order_list.get(sample_product, 0)
            self.order_list[sample_product] = num+1

    def __str__(self):
        return ',\n'.join([k + ': ' + str(v) for (k, v) in self.order_list.items()])

    def __delitem__(self, key):
        del self.order_list[key]

    def _gather_products(self):
        return {k: len(v) for (k, v) in self.product_scheme.items()}

    def _sample_product(self):
        if len(self.list_of_products) == 0:
            return 0
        indices = np.arange(len(self.list_of_products.keys()))
        idx = np.random.choice(indices)
        prod_name, num = [(k, v) for (k, v) in self.list_of_products.items()][idx]
        if num > 1:
            self.list_of_products[prod_name] -= 1
        else:
            del self.list_of_products[prod_name]
        return prod_name

    def pop(self, key):
        self.order_list[key] -= 1
        if self.order_list[key] == 0:
            del self.order_list[key]

    def get(self, key, default):
        return self.order_list.get(key, default)
