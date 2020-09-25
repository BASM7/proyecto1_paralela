import sys
import getopt
import numpy as np
from worm import Worm
from mpi4py import MPI


def load_data(filename):
    f = open(filename, "r")
    filas = f.readlines()
    print(filas)
    loaded_data = np.zeros(shape=(len(filas), 10))
    for i, fila in enumerate(filas):
        tlist = [int(s) for s in fila.__str__().split(',')]
        alist = np.array(tlist[0:-1])
        loaded_data[i] = alist
    return loaded_data


def get_command_line_values(argv):
    r_str = ""
    g_str = ""
    s_str = ""
    i_str = ""
    l_str = ""

    try:
        opts, args = getopt.getopt(argv, "r:g:s:i:l", ["R=", "G=", "S=", "I=", "L="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-r", "--R"):
            r_str = arg
        elif opt in ("-g", "--G"):
            g_str = arg
        elif opt in ("-s", "--S"):
            s_str = arg
        elif opt in ("-i", "--I"):
            i_str = arg
        elif opt in ("-l", "--L"):
            l_str = arg

    return int(r_str), int(g_str), int(s_str), int(i_str), int(l_str)


def euclidean_distance(v, u):
    return np.linalg.norm(v - u)


def main():
    list_data = load_data("poker-hand-training-true.data")

    # print(list_data)

    first_data = list_data[0]
    for i, data in enumerate(list_data):
        print(i, ' : ', euclidean_distance(first_data, data))

    # a = np.array([1, 1, 4, 10, 4, 6, 2, 3, 3, 4])
    # b = np.array([4, 1, 4, 10, 2, 6, 4, 3, 2, 4])
    # c = np.array([1, 6, 4, 1, 4, 11, 2, 2, 3, 5])


if __name__ == '__main__':
    main()
