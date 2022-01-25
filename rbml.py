import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


class RBML:
    """
    The Default constructor method.
    :param a: alfa value
    :param b: beta value
    :param k: nearest neigbour value
    :param dataset: name of the dataset
    """
    def __init__(self, a=0.5, b=3, k_neighbors=3, dataset='iris'):
        self.dataset = dataset
        self.b = b
        self.a = a
        self.k_neighbors = k_neighbors
        self.x = None
        self.y = None
        self.distance_matrix = None
        self.avg_margins = []

    def fit_transform(self, x, y, iteration=1):
        self.x = x
        self.y = y
        for it in range(1, iteration + 1):
            # Distance matrix for all xi points
            self.distance_matrix = self.calculate_distance_matrix(self.x)

            x_stars = []
            margins = []
            for i in range(len(x)):
                mi = self.calculate_mi(i, self.distance_matrix[i])
                margins.append(mi)
                xn = self.calculate_xn(i, self.distance_matrix[i])
                x_impostor = self.calculate_impostor_mean(mi, self.distance_matrix[i], i)
                if np.isnan(x_impostor).any():
                    x_h = x[i].copy()
                else:
                    x_h = x_impostor + mi * (x[i] - x_impostor) / np.linalg.norm(x[i] - x_impostor)

                xi_new = (1 - self.a) * xn + self.a * x_h
                x_stars.append(xi_new)

            self.avg_margins.append(np.array(margins).mean())
            self.x = np.array(x_stars)
            #print(f'Iteration {it}\t\tAvg margin: {self.avg_margins[-1]:.3f}')

        return self.x

    def drag_t(self, i, j):
        vector = self.distance_matrix[i]
        vector[i] = np.inf
        vector = np.where(self.y == self.y[i], vector, np.inf)
        k_nearest = np.argsort(vector)[:self.k_neighbors]
        return j in k_nearest

    def calculate_mi(self, i, distance_vector):
        # filter only same class labels. others will be 0.
        vector = np.array([val if self.drag_t(i, j) else 0 for j, val in enumerate(distance_vector)])
        # get furthest point which respect to i.th point.
        furthest_point = np.argsort(vector)[-1]
        return self.b * vector[furthest_point]

    @staticmethod
    def calculate_distance_matrix(x):
        return distance.squareform(distance.pdist(x))

    def calculate_xn(self, i, distance_vector):
        k_neighbors = self.calculate_target_neighbors(i, distance_vector)
        result = np.mean(k_neighbors, axis=0)
        return result

    def calculate_impostor_mean(self, mi, distance_vector, i):
        vector = np.where(distance_vector < mi, distance_vector, np.nan)
        vector = np.where(self.y != self.y[i], vector, np.nan)
        if np.isnan(vector).all():
            return [np.nan]
        else:
            indicies = ~np.isnan(vector)
            impostors = self.x[indicies]
            impostor_mean = impostors.mean(axis=0)
            return impostor_mean

    def calculate_target_neighbors(self, i, distance_vector):
        vector = np.array([val if self.drag_t(i, j) else 0 for j, val in enumerate(distance_vector)])
        k_neighbors = np.argsort(vector)[-self.k_neighbors:]
        result = self.x[k_neighbors]
        return result
