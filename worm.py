import numpy as np


class Worm:
    __slots__ = ['luciferin', 'radius', 'fitness', 'internal_distance',
                 'neighbors_worms', 'covered_data', 'position']

    def __init__(self, new_luciferin, new_position):
        self.luciferin = new_luciferin
        self.fitness = 0.0
        self.internal_distance = 0.0
        self.neighbors_worms = None
        self.covered_data = None
        self.position = new_position

    def sum_luciferin(self):
        sum_total = 0
        for worm in self.neighbors_worms:
            sum_total += worm.luciferin
        return sum_total

    def __len__(self):
        return len(self.covered_data) if self.covered_data is not None else 0


def main():
    pass


if __name__ == '__main__':
    main()
