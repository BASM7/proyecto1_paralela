import numpy as np


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def calculate_intra_distance(worm):
    sum_total = 0
    if worm.covered_data is not None and len(worm.covered_data) > 0:
        sum_total = sum([euclidean_distance(data_point, worm.position) for data_point in worm.covered_data])
    return sum_total


def calculate_sum_squared_errors(centroids_list):
    sum_total = 0
    for worm in centroids_list:
        worm_pos = worm.position
        sum_temp = 0
        if worm.covered_data is not None:
            for data in worm.covered_data:
                sum_temp += (euclidean_distance(worm_pos, data) ** 2)
        sum_total += sum_temp
    return sum_total


def calculate_inter_centroid_distance(centroid_list):
    sum_total = 0
    for centroid in centroid_list:
        for second_centroid in centroid_list:
            sum_total += (euclidean_distance(centroid.position, second_centroid.position) ** 2)
    return sum_total


def calculate_max_internal_distance(centroid_list):
    max_internal_dist = 0
    for worm in centroid_list:
        if worm.internal_distance > max_internal_dist:
            max_internal_dist = worm.internal_distance
    return max_internal_dist


def calculate_fitness(worm, sum_squared_errors, max_internal_dist, cant_data, inter_dist):
    nominator = inter_dist * (1 / cant_data) * len(worm)
    denominator = sum_squared_errors * (worm.internal_distance / max_internal_dist)
    new_fitness = nominator / denominator
    return new_fitness


def calculate_luciferin(worm, constant_decay, enhancement_fraction):
    return (1 - constant_decay) * worm.luciferin + enhancement_fraction * worm.fitness


def calculate_probability(worm1, worm2):
    pass


def calculate_new_position(worm, brightest_neighbor, worm_step):
    new_position = worm.position
    if brightest_neighbor is not None:
        distance = euclidean_distance(worm.position, brightest_neighbor.position)
        if distance != 0:
            difference = np.subtract(brightest_neighbor.position, worm.position)
            fraction = worm_step / distance
            summand = np.multiply(difference, fraction)
            new_position = np.add(worm.position, summand)
    return new_position


# def calculate_new_position(worm, brightest_neighbor, worm_step):
#     brightest_neighbor: Worm
#     new_position = worm.position
def main():
    array = np.array([2.3, 4.5])
    print(np.multiply(array, (0.03 / 3.1)))


if __name__ == '__main__':
    main()
