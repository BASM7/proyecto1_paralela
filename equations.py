import numpy as np


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def calculate_intra_distance(worm):
    sum_total = 0
    if worm.covered_data is not None:
        sum_total = sum([euclidean_distance(data_point, worm.position) for data_point in worm.covered_data])
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
    for centroid_index in centroid_list:
        for second_centroid_index in centroid_list:
            sum_total += (euclidean_distance(swarm[centroid_index].position,
                                             swarm[second_centroid_index].position) ** 2)
    return sum_total


def calculate_max_internal_distance(swarm, centroid_list):
    max_internal_dist = 0
    for index in centroid_list:
        worm = swarm[index]
        if worm.internal_distance > max_internal_dist:
            max_internal_dist = worm.internal_distance
    return max_internal_dist


def calculate_fitness(worm, sum_squared_errors, max_internal_dist, cant_data):
    nominator = (1 / cant_data) * len(worm)
    demoninator = sum_squared_errors * (worm.internal_distance / max_internal_dist)
    return nominator / demoninator


def calculate_luciferin(previous_luciferin, luciferin_constant_decay, luciferin_enhancement_fraction):
    pass


def main():
    array1 = np.array([2.6006387, 8.22735151, 2.59879434, 5.44134471, 1.74145815, 10.56438908,
                       3.58889295, 5.43374375, 1.82441594, 10.69579439])

    array2 = np.array([1.76302288, 8.45526787, 2.08559538, 7.68420067, 2.9355501, 10.92414194,
                       3.48622029, 5.16347038, 1.55036828, 9.63558866])

    array3 = np.array([1.29230881, 10.04953025, 2.3225288, 11.41011835, 1.55839111, 1.20803691,
                       2.98409638, 10.36286252, 1.9893777, 5.91160813])

    array4 = np.array([1.31275648, 12.49244057, 1.2915126, 8.13393705, 1.373989, 10.15408052,
                       1.57791889, 4.22384385, 2.18566904, 3.9706235])

    array5 = np.array([3.86327776, 1.39613987, 1.75567472, 3.45147908, 2.18981997, 8.01820005,
                       2.33965478, 10.66301395, 2.74581319, 7.57300941])

    print(euclidean_distance(array1, array2))
    # print(euclidean_distance(array1, array3))
    # print(euclidean_distance(array1, array4))
    # print(euclidean_distance(array3, array4))
    # print(euclidean_distance(array2, array4))


if __name__ == '__main__':
    main()
