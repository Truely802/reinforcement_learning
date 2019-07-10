class Product(object):
    def __init__(self, name, weight, volume, manufacturer, price):  # , count=1):
        self.name = name
        self.weight = weight
        self.volume = volume
        self.manufacturer = manufacturer
        self.price = price
        # self.count = count


class Shelf(object):

    def __init__(self, coordinates, max_volume, max_weight):  # , max_floors):
        self.coordinates = coordinates
        self.max_volume = max_volume  # maximum volume in liters per 1 section
        self.max_weight = max_weight  # maximum weight in kilos
        # self.n_sections = n_sections  # number of sections in length
        # self.max_floors = max_floors  # maximum floors on shelf
        self.products = dict()
        self.free_volume = max_volume
        self.available_load = max_weight
        self.origin = "shelf"
        self.sprite = "#"
        self.passable = False

    def put_product(self, product):
        if self.free_volume < product.volume and self.available_load < product.weight:
            print('The shelf is overburden.')
            return 0
        self.products[product.name] = self.products.get(product.name, {'product': product,  'count': 0})
        self.free_volume -= product.volume
        self.available_load -= product.weight
        self.products[product.name]['count'] += 1
        return 1

    def remove_product(self, product_name):
        if product_name not in self.products or self.products[product_name]['count'] == 0:
            print('No such a product.')
            return 0
        self.free_volume += self.products[product_name]['product'].volume
        self.available_load += self.products[product_name]['product'].weight
        self.products[product_name]['count'] -= 1
        return self.products[product_name]['product']

    def inspect(self):
        if len(self.products) == 0:
            print('The shelf is empty.')
            return 0
        return sorted([(k, v['count']) for (k, v) in self.products.items()], reverse=True, key=lambda x: x[1])


class Agent(object):

    def __init__(self, name='Bill', max_volume=30, max_weight=30, coordinates=(1,1)):
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

    def move(self, map_obj, to='u'):
        if to == 'u':
            dest = (self.coordinates[0], self.coordinates[1] + 1)
        elif to == 'd':
            dest = (self.coordinates[0], self.coordinates[1] - 1)
        elif to == 'l':
            dest = (self.coordinates[0] - 1, self.coordinates[1])
        elif to == 'r':
            dest = (self.coordinates[0] + 1, self.coordinates[1])
        else:
            print('Wrong destination code.')
            return 0

        if not map_obj[dest[0]][dest[1]].passable:
            print("Can't move here.")
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
            print('No such a product.')
            return 0
        self.free_volume -= self.products[product_name]['product'].volume
        self.available_load -= self.products[product_name]['product'].weight
        self.inventory[product_name]['count'] -= 1
        product = self.inventory[product_name]['product']
        response = shelf.put_product(product)
        return response

    def take_product(self, product_name):
        shelf = self._find_shelf()
        product = shelf.remove_product(product_name)
        if product == 0:
            return 0
        if self.free_volume < product.volume and self.available_load < product.weight:
            print('No more space in inventory')
            self.put_product(product_name=product.name)
            return 0
        self.inventory[product.name] = self.inventory.get(product.name, {'product': product, 'count': 0})
        self.free_volume -= product.volume
        self.available_load -= product.weight
        self.inventory[product.name]['count'] += 1
        return 1

    def inspect_shelf(self, map_obj):
        shelf = self._find_shelf(map_obj)
        return shelf.inspect()


class PickPoint(Shelf):

    def __init__(self, coordinates):
        super().__init__(coordinates, max_volume=None, max_weight=None)
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

    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.passable = True
        self.origin = "floor"
        self.sprite = "."


class Wall(object):

    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.passable = False
        self.origin = "wall"
        self.sprite = "+"
