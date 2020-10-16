"""
P 1.
Autores:
Jesús Alonso Moreno Montero B95346
Luis Alfonso Jiménez Aguilar B93986
"""

import sys
import getopt
import os
import pdb
from equations import *
from worm import Worm
from mpi4py import MPI

# FILE_DATA = "poker-hand-training-true.data"
FILE_DATA = "test.data"
# FILE_DATA = "mini_test.data"
# FILE_DATA = "tiny_mini_test.data"

LEN = 1
FIT = 2


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


def build_kdimentional_tree_of_swarm(swarm, depth=0):
    nodes = len(swarm)
    if nodes <= 0:
        return None
    splitting_axis = depth % 10
    sorted_points = sorted(swarm, key=lambda worm: worm.position[splitting_axis])
    return {
        'node': sorted_points[nodes // 2],
        'left': build_kdimentional_tree_of_swarm(sorted_points[:nodes // 2], depth + 1),
        'right': build_kdimentional_tree_of_swarm(sorted_points[nodes // 2 + 1:], depth + 1)
    }


def closer_distance(pivot, p1, p2):
    if p1 is None:
        return p2

    if p2 is None:
        return p1

    distance1 = euclidean_distance(pivot, p1)
    distance2 = euclidean_distance(pivot, p2)

    return p1 if distance1 < distance2 else p2


# def kdimentional_tree_closest_point(root, data_point, depth=0):
#     if root is None:
#         return None
#     spliting_axis = depth % 10
#     if data_point[spliting_axis] < root['node'][spliting_axis]:
#         next_subtree = root['left']
#         opposite_subtree = root['right']
#     else:
#         next_subtree = root['right']
#         opposite_subtree = root['left']
#     best_result = closer_distance(data_point,
#                                   kdimentional_tree_closest_point(next_subtree, data_point, depth + 1),
#                                   root['node'])
#     if euclidean_distance(data_point, best_result) > abs(data_point[spliting_axis] - root['node'][spliting_axis]):
#         best_result = closer_distance(data_point,
#                                       kdimentional_tree_closest_point(opposite_subtree, data_point, depth + 1),
#                                       best_result)
#     return best_result


def create_swarm(list_data, starting_luminicesce, radius):
    swarm = []
    min_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    max_values = [4, 13, 4, 13, 4, 13, 4, 13, 4, 13]
    for i in range(0, int(len(list_data) * 0.9)):
        uniform_data_point = np.random.uniform(low=min_values, high=max_values, size=10)
        new_glowworm = Worm(starting_luminicesce, uniform_data_point, radius)
        swarm.append(new_glowworm)
    return swarm


def get_neighborhood(root, worm, depth=0, neighbors_worms=None):
    if neighbors_worms is None:
        neighbors_worms = []
    if root is not None:
        distance = euclidean_distance(root['node'].position, worm.position)
        if (distance < worm.radius) and (root['node'].luciferin > worm.luciferin):
            neighbors_worms.append(root['node'])

        left_tree = root['left']
        right_tree = root['right']

        splitting_axis = depth % 10

        if worm.position[splitting_axis] < root['node'].position[splitting_axis]:
            get_neighborhood(left_tree, worm, depth + 1, neighbors_worms)
            opposite_subtree = right_tree
        else:
            get_neighborhood(right_tree, worm, depth + 1, neighbors_worms)
            opposite_subtree = left_tree
        if opposite_subtree is not None:
            child_worm = opposite_subtree['node']
            if abs(child_worm.position[splitting_axis] - worm.position[splitting_axis]) <= worm.radius:
                get_neighborhood(opposite_subtree, worm, depth + 1, neighbors_worms)
    return neighbors_worms


def get_covered_data(root, data_point, radius, depth=0, list_data=None):
    if list_data is None:
        list_data = []
    if root is not None:
        if euclidean_distance(root['node'], data_point) < radius:
            list_data.append(root['node'])

        left_tree = root['left']
        right_tree = root['right']

        splitting_axis = depth % 10

        if data_point[splitting_axis] < root['node'][splitting_axis]:
            get_covered_data(left_tree, data_point, radius, depth + 1, list_data)
            opposite_subtree = right_tree
        else:
            get_covered_data(right_tree, data_point, radius, depth + 1, list_data)
            opposite_subtree = left_tree
        if opposite_subtree is not None:
            if abs(opposite_subtree['node'][splitting_axis] - data_point[splitting_axis]) <= radius:
                get_covered_data(opposite_subtree, data_point, radius, depth + 1, list_data)
    return list_data


def get_centroid_list(swarm):
    centroid_list = []
    for worm in swarm:
        if not centroid_list:
            centroid_list.append(worm)
        else:
            valid = True
            for temp_worm in centroid_list:
                distance = euclidean_distance(temp_worm.position, worm.position)
                if distance < worm.radius:
                    valid = False
                    break
            if valid:
                centroid_list.append(worm)
    return centroid_list


def start_covered_data(tree, worm):
    return get_covered_data(tree, worm.position, worm.radius)


def clean_swarm(swarm):
    new_swarm = []
    for worm in swarm:
        if worm.covered_data is not None and len(worm.covered_data) > 0:
            new_swarm.append(worm)
    return new_swarm


def sort_swarm(swarm, criterion):
    if criterion == LEN:
        return sorted(swarm, key=lambda x: len(x), reverse=True)
    else:
        return sorted(swarm, key=lambda x: x.fitness, reverse=True)


def get_tree_size(root):
    size = 0
    if root is not None:
        size += 1
    else:
        return size

    size += get_tree_size(root['left'])
    size += get_tree_size(root['right'])

    return size


def get_min_index(rank, collection, size):
    return int((rank * len(collection) / size))


def get_max_index(rank, collection, size):
    return int((rank + 1) * len(collection) / size)


def set_covered_data(swarm, tree):
    for worm in swarm:
        worm.covered_data = np.array(start_covered_data(tree, worm))
        worm.internal_distance = calculate_intra_distance(worm)
    return swarm


def set_worm_neighborhood(swarm, tree_swarm):
    for worm in swarm:
        worm.neighbors_worms = get_neighborhood(tree_swarm, worm)
    return swarm


def update_luciferin_fitness(swarm, sum_squared_errors,
                             max_internal_dist, cant_data, lucifering_dec, lucifering_inc, inter_dist):
    for worm in swarm:
        worm.fitness = calculate_fitness(worm, sum_squared_errors, max_internal_dist, cant_data, inter_dist)
        new_luciferin = calculate_luciferin(worm, lucifering_dec, lucifering_inc)
        worm.luciferin = new_luciferin
    return swarm


def update_positions_and_data(swarm, worm_step, tree):
    for worm in swarm:
        previous_pos = worm.position
        worm.position = calculate_new_position(worm, worm.get_brightest_neighbor(), worm_step)
        if np.all(previous_pos != worm.position):
            worm.covered_data = np.array(start_covered_data(tree, worm))
            worm.internal_distance = calculate_intra_distance(worm)
    return swarm


def print_list(collection):
    for item in collection:
        print(item)


def record_output(new_line):
    output_file = open("salida.txt", "a")
    output_file.write(new_line + '\n')
    output_file.close()


def record_time(total_time):
    time_file = open("time.txt", "w")
    time_file.write(str(total_time) + ' segundos')
    time_file.close()


def main(argv):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    list_data = []
    swarm = []
    tree = {}

    lucifering_dec = 0.0
    lucifering_inc = 0.0
    worm_step = 0.0
    # radius = 0.0
    # starting_luciferin = 0.0

    t_start = MPI.Wtime()

    if rank == 0:
        # delete the exit file if it exists.
        if os.path.exists('salida.txt'):
            os.remove('salida.txt')

        lucifering_dec, lucifering_inc, worm_step, radius, starting_luciferin = \
            get_command_line_values(argv)
        list_data = load_data(FILE_DATA)
        tree = build_kdimentional_tree(list_data)

        # TODO: paralelizar creacion de enjambre.
        swarm = create_swarm(list_data, starting_luciferin, radius)

    list_data, tree, swarm, lucifering_dec, lucifering_inc, worm_step = \
        comm.bcast((list_data, tree, swarm, lucifering_dec, lucifering_inc, worm_step), root=0)

    if rank == 0:
        swarm_chunk = [[] for _ in range(size)]
        for i, worm in enumerate(swarm):
            swarm_chunk[i % size].append(worm)
    else:
        swarm_chunk = None

    # Etapa de inicializacion.

    swarm = comm.scatter(swarm_chunk, root=0)
    swarm = set_covered_data(swarm, tree)
    new_swarm = comm.gather(swarm, root=0)
    if rank == 0:
        swarm = [worm for swarm_chunk in new_swarm for worm in swarm_chunk]
        swarm = sort_swarm(clean_swarm(swarm), LEN)
        list_centroid_candidates = get_centroid_list(swarm)
        sse = calculate_sum_squared_errors(list_centroid_candidates)

        record_output('cantidad inicial de centroides: ' + str(len(list_centroid_candidates)))
        # pdb.set_trace()
    else:
        list_centroid_candidates = None
        sse = None
        swarm = None

    iteration = 0
    # len(list_centroid_candidates) > 10 and
    while iteration < 3:

        comm.Barrier()
        if rank == 0:
            max_internal_dist = calculate_max_internal_distance(list_centroid_candidates)
            inter_dist = calculate_inter_centroid_distance(list_centroid_candidates)
        else:
            max_internal_dist = None
            inter_dist = None

        max_internal_dist, inter_dist, sse = comm.bcast((max_internal_dist, inter_dist, sse), root=0)

        # Romper la lista de gusanos en pezados.
        comm.Barrier()
        if rank == 0:
            # print(swarm[0])
            swarm_chunk = [[] for _ in range(size)]
            for i, worm in enumerate(swarm):
                swarm_chunk[i % size].append(worm)
        else:
            swarm_chunk = None

        comm.Barrier()

        swarm = comm.scatter(swarm_chunk, root=0)
        swarm = update_luciferin_fitness(swarm, sse, max_internal_dist, len(list_data),
                                         lucifering_dec, lucifering_inc, inter_dist)
        new_swarm = comm.gather(swarm, root=0)

        comm.Barrier()
        if rank == 0:
            swarm = [worm for swarm_chunk in new_swarm for worm in swarm_chunk]
            tree_swarm = build_kdimentional_tree_of_swarm(swarm)
        else:
            tree_swarm = None
            swarm = None

        tree_swarm = comm.bcast(tree_swarm, root=0)

        comm.Barrier()
        if rank == 0:
            swarm_chunk = [[] for _ in range(size)]
            for i, worm in enumerate(swarm):
                swarm_chunk[i % size].append(worm)
        else:
            swarm_chunk = None

        comm.Barrier()

        swarm = comm.scatter(swarm_chunk, root=0)
        swarm = set_worm_neighborhood(swarm, tree_swarm)
        swarm = update_positions_and_data(swarm, worm_step, tree)
        new_swarm = comm.gather(swarm, root=0)

        comm.Barrier()
        if rank == 0:
            # nuevo ordenamiento con el fitness.
            swarm = [worm for swarm_chunk in new_swarm for worm in swarm_chunk]
            swarm = sort_swarm(clean_swarm(swarm), FIT)
            list_centroid_candidates = get_centroid_list(swarm)
            sse = calculate_sum_squared_errors(list_centroid_candidates)
            iteration += 1

            record_output('i: ' + str(iteration) + ' cantidad de centroides: ' + str(len(list_centroid_candidates)))
        else:
            swarm = None
            list_centroid_candidates = None
            sse = None

        # comm.Barrier()
        iteration = comm.bcast(iteration, root=0)

    t_final = MPI.Wtime()
    total_time = comm.allreduce(t_final - t_start, op=MPI.MAX)

    if rank == 0:
        # print('cant worms: ', len(swarm))
        # for i, worm in enumerate(swarm):
        #     print(i, ' -> ', worm.fitness)
        record_time(total_time)

    return


if __name__ == '__main__':
    main(sys.argv[1:])

# mpiexec -n 4 python main.py -r 0.4 -g 0.6 -s 0.03 -i 4.0 -l 5.0

