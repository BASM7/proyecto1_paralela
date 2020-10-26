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
from scipy.spatial import KDTree

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
    loaded_data = np.empty(shape=(len(filas), 10), dtype=int)
    for i, fila in enumerate(filas):
        tlist = [int(s) for s in fila.__str__().split(',')]
        alist = np.array(tlist[0:-1])
        loaded_data[i] = alist
    return loaded_data


def create_point():
    min_values = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    max_values = np.array([4, 13, 4, 13, 4, 13, 4, 13, 4, 13])
    return np.random.uniform(low=min_values, high=max_values, size=10)


def set_covered_data(local_swarm, tree_data, list_data, radius):
    for worm in local_swarm:
        worm.covered_data = tree_data.query_ball_point(worm.position, radius)
        worm.internal_distance = calculate_intra_distance(worm, list_data)
    return local_swarm


def pick_centroid(global_swarm, position, centroids):
    centroids_positions = []
    for centroid in centroids:
        centroids_positions.append(global_swarm[centroid].position)
    centroids_positions = np.array(centroids_positions)
    tree_centroid = KDTree(centroids_positions)
    centroid_picked = centroids[tree_centroid.query(position, k=2)[1][1]]
    # pdb.set_trace()
    return centroid_picked


def set_worm_neighborhood(global_swarm, local_swarm, tree_swarm, centroids, radius):
    for worm in local_swarm:
        worm.neighbors_worms = tree_swarm.query_ball_point(worm.position, radius)
        real_neighbors = []
        for index in worm.neighbors_worms:
            if global_swarm[index].luciferin > worm.luciferin:
                real_neighbors.append(index)
        if len(real_neighbors) == 0:
            real_neighbors.append(pick_centroid(global_swarm, worm.position, centroids))
            # pdb.set_trace()
        worm.neighbors_worms = np.array(real_neighbors, dtype=int)
    return local_swarm
    pass


def update_fitness(local_swarm, sse, max_inter_dist, cant_data, inter_dist):
    for worm in local_swarm:
        worm.fitness = calculate_fitness(len(worm), worm.internal_distance, sse, max_inter_dist, cant_data)
    return local_swarm


def update_luciferin(local_swarm, luci_dec, luci_inc):
    for worm in local_swarm:
        worm.luciferin = calculate_luciferin(worm, luci_dec, luci_inc)
    return local_swarm


def roulette_selection(probabilites):
    weight_sum = sum(probabilites)
    value = np.random.random() * weight_sum
    for index, probability in enumerate(probabilites):
        value -= probability
        if value <= 0:
            return index
    return len(probabilites) - 1


def get_brightest_neighbor(global_swarm, worm):
    probabilities = calculate_probability_neighbors(global_swarm, worm)
    chosen_index = roulette_selection(probabilities)
    if worm.neighbors_worms is not None and len(probabilities) > 0:
        return global_swarm[worm.neighbors_worms[chosen_index]]
    return None


def update_positions_and_data(global_swarm, local_swarm, list_data, worm_step, tree_data, radius):
    for worm in local_swarm:
        b_neighbor = get_brightest_neighbor(global_swarm, worm)
        if b_neighbor is not None:
            worm.position = calculate_new_position(worm.position, b_neighbor.position, worm_step)
            worm.covered_data = tree_data.query_ball_point(worm.position, radius)
            worm.internal_distance = calculate_intra_distance(worm, list_data)
    return local_swarm


def get_centroid_list(global_swarm, radius):
    centroid_list = []
    for index, worm in enumerate(global_swarm):
        if not centroid_list:
            centroid_list.append(index)
        else:
            valid = True
            for temp_index in centroid_list:
                temp_worm = global_swarm[temp_index]
                distance = euclidean_distance(temp_worm.position, worm.position)
                if distance < radius:
                    valid = False
                    break
            if valid:
                centroid_list.append(index)
    return np.array(centroid_list, dtype=int)


def clean_swarm(global_swarm):
    new_swarm = []
    for worm in global_swarm:
        if len(worm) > 0:
            new_swarm.append(worm)
    return new_swarm


def sort_swarm(global_swarm, criterion):
    if criterion == LEN:
        return sorted(global_swarm, key=lambda x: len(x), reverse=True)
    else:
        return sorted(global_swarm, key=lambda x: x.fitness, reverse=True)


def get_swarm_chunks(global_swarm, size):
    swarm_chunk = [[] for _ in range(size)]
    for i, worm in enumerate(global_swarm):
        swarm_chunk[i % size].append(worm)
    return swarm_chunk


def record_output(new_line):
    output_file = open("salida.txt", "a")
    output_file.write(new_line + '\n')
    output_file.close()


def record_time(total_time):
    time_file = open("time.txt", "w")
    time_file.write(str(total_time) + ' segundos')
    time_file.close()


def main(argv):
    FILE_DATA = "poker-hand-training-true.data"
    # FILE_DATA = "test.data"
    # FILE_DATA = "mini_test.data"
    # FILE_DATA = "tiny_mini_test.data"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    iteration = 0

    starting_luciferin = 0.0
    luci_dec = 0.0
    luci_inc = 0.0
    worm_step = 0.0
    radius = 0.0

    t_start = MPI.Wtime()

    if rank == 0:
        if os.path.exists('salida.txt'):
            os.remove('salida.txt')

        luci_dec, luci_inc, worm_step, radius, starting_luciferin = get_command_line_values(argv)

        # menu for debugging, TODO: remove for kabré.
        # print("1) poker-hand-training-true.data")
        # print("2) test.data")
        # print("3) mini_test.data")
        # print("4) tiny_mini_test.data")
        # op = input("Data? ")
        #
        # if op == "1":
        #     FILE_DATA = "poker-hand-training-true.data"
        # elif op == "2":
        #     FILE_DATA = "test.data"
        # elif op == "3":
        #     FILE_DATA = "mini_test.data"
        # elif op == "4":
        #     FILE_DATA = "tiny_mini_test.data"

        list_data = load_data(FILE_DATA)

    else:
        list_data = None

    list_data = comm.bcast(list_data, root=0)
    starting_luciferin, luci_dec, luci_inc = comm.bcast((starting_luciferin, luci_dec, luci_inc), root=0)
    worm_step, radius = comm.bcast((worm_step, radius), root=0)

    min_n = int(rank * (len(list_data) * 0.04) / size)
    max_n = int((rank + 1) * (len(list_data) * 0.04) / size)

    local_swarm = []
    for index in range(min_n, max_n):
        worm = Worm(starting_luciferin, create_point())
        local_swarm.append(worm)

    global_swarm = comm.gather(local_swarm, root=0)

    if rank == 0:
        global_swarm = [worm for local_swarm in global_swarm for worm in local_swarm]
        tree_data = KDTree(list_data)
    else:
        tree_data = None

    tree_data, global_swarm = comm.bcast((tree_data, global_swarm), root=0)

    # SET UP PHASE

    if rank == 0:
        swarm_chunk = get_swarm_chunks(global_swarm, size)
    else:
        swarm_chunk = None

    local_swarm = comm.scatter(swarm_chunk, root=0)
    local_swarm = set_covered_data(local_swarm, tree_data, list_data, radius)
    local_swarm = comm.gather(local_swarm, root=0)

    if rank == 0:
        global_swarm = [worm for swarm_chunk in local_swarm for worm in swarm_chunk]

        global_swarm = sort_swarm(clean_swarm(global_swarm), LEN)
        list_centroids = get_centroid_list(global_swarm, radius)
        sse = calculate_sum_squared_errors(global_swarm, list_centroids)
        terminal_condition = len(list_centroids)
        record_output('cantidad inicial de centroides: ' + str(terminal_condition))
    else:
        list_centroids = None
        sse = 0.0
        global_swarm = None
        terminal_condition = 0

    terminal_condition = comm.bcast(terminal_condition, root=0)
    # while iteration < 1:
    while terminal_condition > 10:
        if rank == 0:
            max_internal_dist = calculate_max_internal_distance(global_swarm, list_centroids)
            inter_dist = calculate_inter_centroid_distance(global_swarm, list_centroids)
        else:
            max_internal_dist = 0.0
            inter_dist = 0.0

        list_centroids = comm.bcast(list_centroids, root=0)
        max_internal_dist = comm.bcast(max_internal_dist, root=0)
        inter_dist = comm.bcast(inter_dist, root=0)
        sse = comm.bcast(sse, root=0)

        if rank == 0:
            swarm_chunk = get_swarm_chunks(global_swarm, size)
        else:
            swarm_chunk = None

        local_swarm = comm.scatter(swarm_chunk, root=0)
        local_swarm = update_fitness(local_swarm, sse, max_internal_dist, len(list_data), inter_dist)
        local_swarm = update_luciferin(local_swarm, luci_dec, luci_inc)
        local_swarm = comm.gather(local_swarm, root=0)

        if rank == 0:
            global_swarm = [worm for swarm_chunk in local_swarm for worm in swarm_chunk]
            worm_positions = [worm.position for worm in global_swarm]
            tree_swarm = KDTree(worm_positions)
        else:
            tree_swarm = None
            global_swarm = None

        global_swarm = comm.bcast(global_swarm, root=0)
        tree_swarm = comm.bcast(tree_swarm, root=0)

        if rank == 0:
            swarm_chunk = get_swarm_chunks(global_swarm, size)
        else:
            swarm_chunk = None

        local_swarm = comm.scatter(swarm_chunk, root=0)
        local_swarm = set_worm_neighborhood(global_swarm, local_swarm, tree_swarm, list_centroids, radius)
        local_swarm = update_positions_and_data(global_swarm, local_swarm, list_data, worm_step, tree_data, radius)
        local_swarm = comm.gather(local_swarm, root=0)

        if rank == 0:
            global_swarm = [worm for swarm_chunk in local_swarm for worm in swarm_chunk]
            global_swarm = sort_swarm(clean_swarm(global_swarm), FIT)
            list_centroids = get_centroid_list(global_swarm, radius)
            sse = calculate_sum_squared_errors(global_swarm, list_centroids)

            iteration += 1
            terminal_condition = len(list_centroids)

            # pdb.set_trace()

            record_output('i: ' + str(iteration) + ' cantidad de centroides: ' + str(len(list_centroids)))
        else:
            global_swarm = None
            list_centroids = None
            sse = None

        # iteration = comm.bcast(iteration, root=0)
        terminal_condition = comm.bcast(terminal_condition, root=0)

    t_final = MPI.Wtime()
    total_time = comm.reduce(t_final - t_start, op=MPI.MAX)

    if rank == 0:
        record_time(total_time)
        for centroid in list_centroids:
            print(centroid, ' : ', global_swarm[centroid].position)

    return


if __name__ == '__main__':
    main(sys.argv[1:])

# mpiexec -n 4 python main.py -r 0.4 -g 0.6 -s 0.03 -i 3.9 -l 5.0
