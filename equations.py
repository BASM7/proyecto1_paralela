import numpy as np


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def update_intra_distance(worm):
    worm.internal_distance = sum([euclidean_distance(data_point, worm.position) for data_point in worm.neighbors])
