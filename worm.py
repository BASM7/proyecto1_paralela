import numpy as np


class Worm:
    def __init__(self, new_luminescence, new_position, new_radius):
        self.luminescence = new_luminescence
        self.radius = new_radius
        self.fitness = 0
        self.internal_distance = 0
        self.neighbors = np.empty([])
        self.position = new_position

    def add_neighbor(self, new_neighbor):
        self.neighbors = np.append(self.neighbors, new_neighbor)

    def __getitem__(self, index):
        return self.position

    def __str__(self):
        return str(self.__class__) + '\n' + '\n'.join(
            ('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))


def main():
    x = np.zeros(10)
    new_worm = Worm(45, x, 4)
    print(new_worm)


if __name__ == '__main__':
    main()
