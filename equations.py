import numpy as np


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
        if worm.covered_data is not None:
            for data in worm.covered_data:
                sum_temp += (euclidean_distance(worm_pos, data) ** 2)
        sum_total += sum_temp
    return sum_total


def calculate_inter_centroid_distance(swarm, centroid_list):
    sum_total = 0
    for index in centroid_list:
        worm = swarm[index]
        for s_index in centroid_list:
            second_worm = swarm[s_index]
            sum_total += (euclidean_distance(worm.position, second_worm.position) ** 2)
    return sum_total


def calculate_max_internal_distance(swarm, centroid_list):
    max_internal_dist = 0
    for index in centroid_list:
        worm = swarm[index]
        if worm.internal_distance > max_internal_dist:
            max_internal_dist = worm.internal_distance
    return max_internal_dist


def calculate_fitness(cant_data_cov, int_dist, sse, max_internal_dist, cant_data, inter_dist):
    nominator = inter_dist * (1 / cant_data) * cant_data_cov
    denominator = sse * (int_dist / max_internal_dist)
    new_fitness = nominator / denominator
    return new_fitness


def calculate_luciferin(worm, constant_decay, enhancement_fraction):
    return (1 - constant_decay) * worm.luciferin + enhancement_fraction * worm.fitness


# def calculate_probability(worm1, worm2):
#     pass


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
    array = np.array([2.3, 4.5])
    print(np.multiply(array, (0.03 / 3.1)))


if __name__ == '__main__':
    main()
