import numpy as np
import math


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def calculate_intra_distance(worm, list_data):
    sum_total = 0
    if len(worm) > 0:
        sum_total = sum([euclidean_distance(list_data[index], worm.position) for index in worm.covered_data])
    return sum_total


def calculate_sum_squared_errors(swarm, centroids_list):
    sum_total = 0
    for index in centroids_list:
        worm = swarm[index]
        worm_pos = worm.position
        sum_temp = 0
        if len(worm) > 0:
            for data in worm.covered_data:
                sum_temp += (math.sqrt(euclidean_distance(worm_pos, data)))
        sum_total += sum_temp
    return sum_total


def calculate_inter_centroid_distance(swarm, centroid_list):
    sum_total = 0
    for index in centroid_list:
        worm = swarm[index]
        for s_index in centroid_list:
            second_worm = swarm[s_index]
            sum_total += (math.sqrt(euclidean_distance(worm.position, second_worm.position)))
    return sum_total


def calculate_max_internal_distance(swarm, centroid_list):
    max_internal_dist = 0
    for index in centroid_list:
        worm = swarm[index]
        if worm.internal_distance > max_internal_dist:
            max_internal_dist = worm.internal_distance
    return max_internal_dist


def calculate_fitness(cant_data_cov, int_dist, sse, max_internal_dist, cant_data, inter_dist=1):
    nominator = inter_dist * (1 / cant_data) * cant_data_cov
    denominator = sse * (int_dist / max_internal_dist)
    new_fitness = nominator / denominator
    return new_fitness


def calculate_luciferin(worm, constant_decay, enhancement_fraction):
    return (1 - constant_decay) * worm.luciferin + enhancement_fraction * worm.fitness


def sum_luciferin(global_swarm, worm):
    sum_luci = 0
    if worm.neighbors_worms is not None:
        for index in worm.neighbors_worms:
            sum_luci += global_swarm[index].luciferin
    return sum_luci


def calculate_probability_neighbors(global_swarm, worm):
    sum_luci = sum_luciferin(global_swarm, worm)
    probabilities = []
    if worm.neighbors_worms is not None:
        for index in worm.neighbors_worms:
            neighbor = global_swarm[index]
            probability = (neighbor.luciferin - worm.luciferin) / (sum_luci - worm.luciferin)
            probabilities.append(probability)
    return probabilities


def calculate_new_position(pos1, pos2, worm_step):
    new_position = pos1
    distance = euclidean_distance(pos1, pos2)
    if distance > 0:
        difference = np.subtract(pos2, pos1)
        fraction = worm_step / distance
        summand = np.multiply(difference, fraction)
        new_position = np.add(pos1, summand)
    return new_position


def main():
    points = np.array([[1, 10, 1, 11, 1, 13, 1, 12, 1, 1], [2, 11, 2, 13, 2, 10, 2, 12, 2, 1],
                       [3, 12, 3, 11, 3, 13, 3, 10, 3, 1], [4, 10, 4, 11, 4, 1, 4, 13, 4, 12],
                       [4, 1, 4, 13, 4, 12, 4, 11, 4, 10]])

    for point in points:
        for second_point in points:
            print(euclidean_distance(point, second_point))


if __name__ == '__main__':
    main()
