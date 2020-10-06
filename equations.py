import numpy as np


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def update_intra_distance(worm):
    worm.internal_distance = sum([euclidean_distance(data_point, worm.position) for data_point in worm.covered_data])


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


def main():
    array1 = np.array([3.09114287, 5.6477811, 2.47177601, 5.11549771, 2.97867931, 1.53942178,
                       2.54899733, 4.03623052, 3.21436392, 5.17299078])
    array2 = np.array([3.04663525, 2.00811928, 2.347577, 5.68641432, 1.70697652, 10.04590724,
                       3.17561016, 6.64526916, 1.17363649, 11.8411182])

    array3 = np.array([1.29230881, 10.04953025, 2.3225288, 11.41011835, 1.55839111, 1.20803691,
                       2.98409638, 10.36286252, 1.9893777, 5.91160813])

    array4 = np.array([1.31275648, 12.49244057, 1.2915126, 8.13393705, 1.373989, 10.15408052,
                       1.57791889, 4.22384385, 2.18566904, 3.9706235])

    array5 = np.array([3.86327776, 1.39613987, 1.75567472, 3.45147908, 2.18981997, 8.01820005,
                       2.33965478, 10.66301395, 2.74581319, 7.57300941])

    print(euclidean_distance(array3, array5))
    print(euclidean_distance(array1, array3))
    print(euclidean_distance(array1, array4))
    print(euclidean_distance(array3, array4))
    print(euclidean_distance(array2, array4))


if __name__ == '__main__':
    main()
