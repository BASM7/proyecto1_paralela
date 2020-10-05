"""
P 1.
Autores:
Jesús Alonso Moreno Montero B95346
Luis Alfonso Jiménez Aguilar B93986
"""

import sys
import getopt
import pprint
from equations import *
from worm import Worm
from mpi4py import MPI

pp = pprint.PrettyPrinter(indent=4)
# FILE_DATA = "poker-hand-training-true.data"
# FILE_DATA = "test.data"
FILE_DATA = "testSwarm.data"


# FILE_DATA = "mini_test.data"


def get_command_line_values(argv):
    r_str = ""
    g_str = ""
    s_str = ""
    i_str = ""
    l_str = ""

    try:
        opts, args = getopt.getopt(argv, "h:r:g:s:i:l:", ["H=", "R=", "G=", "S=", "I=", "L="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-r", "--R"):
            r_str = arg
        elif opt in ("-g", "--G"):
            g_str = arg
        elif opt in ("-s", "--S"):
            s_str = arg
        elif opt in ("-i", "--I"):
            i_str = arg
        elif opt in ("-l", "--L"):
            l_str = arg

    return float(r_str), float(g_str), float(s_str), float(i_str), float(l_str)


def load_data(filename):
    f = open(filename, "r")
    filas = f.readlines()
    loaded_data = np.empty(shape=(len(filas), 10))
    for i, fila in enumerate(filas):
        tlist = [int(s) for s in fila.__str__().split(',')]
        alist = np.array(tlist[0:-1])
        loaded_data[i] = alist
    return loaded_data


def build_kdimentional_tree(list_data, depth=0):
    nodes = len(list_data)
    if nodes <= 0:
        return None
    splitting_axis = depth % 10
    sorted_points = sorted(list_data, key=lambda data: data[splitting_axis])
    return {
        'node': sorted_points[nodes // 2],
        'left': build_kdimentional_tree(sorted_points[:nodes // 2], depth + 1),
        'right': build_kdimentional_tree(sorted_points[nodes // 2 + 1:], depth + 1)
    }


def closer_distance(pivot, p1, p2):
    if p1 is None:
        return p2

    if p2 is None:
        return p1

    distance1 = euclidean_distance(pivot, p1)
    distance2 = euclidean_distance(pivot, p2)

    return p1 if distance1 < distance2 else p2


def kdimentional_tree_closest_point(root, data_point, depth=0):
    if root is None:
        return None
    spliting_axis = depth % 10
    if data_point[spliting_axis] < root['node'][spliting_axis]:
        next_subtree = root['left']
        opposite_subtree = root['right']
    else:
        next_subtree = root['right']
        opposite_subtree = root['left']
    best_result = closer_distance(data_point,
                                  kdimentional_tree_closest_point(next_subtree, data_point, depth + 1),
                                  root['node'])
    if euclidean_distance(data_point, best_result) > abs(data_point[spliting_axis] - root['node'][spliting_axis]):
        best_result = closer_distance(data_point,
                                      kdimentional_tree_closest_point(opposite_subtree, data_point, depth + 1),
                                      best_result)
    return best_result


def create_swarm(list_data, starting_luminicesce, radius):
    swarm = []
    min_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    max_values = [4, 13, 4, 13, 4, 13, 4, 13, 4, 13]
    for i in range(0, int(len(list_data) * 0.9)):
        uniform_data_point = np.random.uniform(low=min_values, high=max_values, size=10)
        new_glowworm = Worm(starting_luminicesce, uniform_data_point, radius)
        swarm.append(new_glowworm)
    return swarm


def get_starting_neighbors(root, data_point, depth=0, list_neighbors=None):
    if list_neighbors is None:
        list_neighbors = []
    if root is not None:
        if euclidean_distance(root['node'], data_point) <= 4:
            list_neighbors.append(root['node'])

        left_tree = root['left']
        right_tree = root['right']

        splitting_axis = depth % 10

        if data_point[splitting_axis] < root['node'][splitting_axis]:
            get_starting_neighbors(left_tree, data_point, depth + 1, list_neighbors)
            opposite_subtree = right_tree
        else:
            get_starting_neighbors(right_tree, data_point, depth + 1, list_neighbors)
            opposite_subtree = left_tree
        if opposite_subtree is not None:
            if abs(opposite_subtree['node'][splitting_axis] - data_point[splitting_axis]) <= 4:
                get_starting_neighbors(opposite_subtree, data_point, depth + 1, list_neighbors)

    return list_neighbors


def start_neighbors(tree, worm):
    worm.neighbors = get_starting_neighbors(tree, worm.position)


def main(argv):
    comm = MPI.COMM_WORLD
    pid = comm.Get_rank()
    size = comm.Get_size()

    list_data = []
    swarm = []
    tree = {}

    lucifering_dec = 0.0
    lucifering_inc = 0.0
    worm_movement_distance = 0.0
    radius = 0.0
    starting_luciferin = 0.0

    if pid == 0:
        lucifering_dec, lucifering_inc, worm_movement_distance, radius, starting_luciferin = \
            get_command_line_values(argv)
        list_data = load_data(FILE_DATA)
        tree = build_kdimentional_tree(list_data)
        swarm = create_swarm(list_data, starting_luciferin, radius)

    t_start = MPI.Wtime()
    list_data, tree, swarm, lucifering_dec, lucifering_inc, worm_movement_distance, radius, starting_luciferin = \
        comm.bcast((
            list_data, tree, swarm, lucifering_dec, lucifering_inc, worm_movement_distance, radius, starting_luciferin),
            0)

    min_index = int((pid * len(swarm) / size))
    max_index = int((pid + 1) * len(swarm) / size)

    # Etapa de inicializacion
    for index in range(min_index, max_index):
        worm = swarm[index]
        # data_point = kdimentional_tree_closest_point(tree, worm.position)
        start_neighbors(tree, worm)
        update_intra_distance(worm)

        print(index, ' : ', worm.position, ' closest neighbors: ')
        for neighbor in worm.neighbors:
            print(neighbor, ' : ', euclidean_distance(neighbor, worm.position))
        print('---')

    # All worms are initialized.

    t_final = MPI.Wtime()
    tw = comm.allreduce(t_final - t_start, op=MPI.MAX)
    if pid == 0:
        pp.pprint(tw)
    return


if __name__ == '__main__':
    main(sys.argv[1:])

# mpiexec -n 4 python main.py -r 0.4 -g 0.6 -s 0.03 -i 4.0 -l 5.0
