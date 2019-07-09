class Product(object):
    def __init__(self, weight, volume, name, manufacturer, price):  # , count=1):
        self.weight = weight
        self.volume = volume
        self.name = name
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

    def __init__(self, name, max_volume, max_weight, coordinates=(0,0)):
        self.name = name
        self.coordinates = coordinates
        self.max_volume = max_volume
        self.max_weight = max_weight
        self.free_volume = max_volume
        self.available_load = max_weight
        self.inventory = dict()
        self.origin = "agent"
        self.sprite = "A"
        self.passable = False

    def move(self, to='u'):
        if to == 'u':
            destination = (self.coordinates[0], self.coordinates[1] + 1)
        elif to == 'd':
            destination = (self.coordinates[0], self.coordinates[1] - 1)
        elif to == 'l':
            destination = (self.coordinates[0] - 1, self.coordinates[1])
        elif to == 'r':
            destination = (self.coordinates[0] + 1, self.coordinates[1])
        else:
            print('Wrong destination code.')
            return 0

        if False:  # TODO condition why cant move to target point
            print("Can't move here.")
            return 0
        self.coordinates = destination
        return 1

    def wait(self):
        return 1

    def _find_shelf(self):
        pass  # TODO write algo to find nearest shelf

    def put_product(self, product_name):
        shelf = self._find_shelf()
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

    def inspect_shelf(self):
        shelf = self._find_shelf()
        return shelf.inspect()