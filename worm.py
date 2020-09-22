class Worm:
    luminescence = 0
    position = 0
    adaptation = 0
    radius = 0

    def __init__(self, new_luminescence, new_position, new_radius):
        self.luminescence = new_luminescence
        self.position = new_position
        self.radius = new_radius

    def __str__(self):
        return str(self.__class__) + '\n' + '\n'.join(
            ('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))


def main():
    """
    Código solo para probar la sobrecarga de print().
    """
    new_worm = Worm(45, [4, 2, 5], 5666)
    print(new_worm)


if __name__ == '__main__':
    main()