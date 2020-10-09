import numpy as np


class Worm:
    def __init__(self, new_luciferin, new_position, new_radius):
        self.luciferin = new_luciferin
        self.radius = new_radius
        self.fitness = 0.0
        self.internal_distance = 0.0
        self.neighbors_worms = []
        self.covered_data = None
        self.position = new_position

    def get_brightest_neighbor(self):
        brightest_neighbor = None
        for worm in self.neighbors_worms:
            if brightest_neighbor is None or brightest_neighbor.luciferin < worm.luciferin:
                brightest_neighbor = worm
        return brightest_neighbor

    def sum_luciferin(self):
        sum_total = 0
        for worm in self.neighbors_worms:
            sum_total += worm.luciferin
        return sum_total

    def __len__(self):
        return len(self.covered_data) if self.covered_data is not None else 0

    def __str__(self):
        return str(self.__class__) + '\n' + '\n'.join(
            ('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))


def main():
    x = np.zeros(10)
    new_worm = Worm(45, x, 4)
    print(new_worm)


if __name__ == '__main__':
    main()
