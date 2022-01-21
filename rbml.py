from sklearn.datasets import load_iris, load_diabetes, load_wine
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class RBML:
    """
    The Default constructor method.
    :param k nearest neigbour value
    :param b beta value
    :param a alfa value
    """

    def __init__(self, k=3, b=3, a=0.3):
        self.b = b
        self.a = a
        self.k = k
        self.x = None
        self.y = None
        self.distance_matrix = None
        self.loo = LeaveOneOut()
        self.kfold = None
        self.mean_mi = []
        self.acc = []

    """
    Training method
    :param x independent features
    :param y dependenet feature
    :param iteration training iteration
    :return transformed/projected training data
    """

    def fit(self, x, y, iteration=1):
        self.x = x
        self.y = y
        # k groups = len(dataset), this is equivalent to the Leave One Out strategy,
        self.kfold = KFold(n_splits=len(self.x) - 50)

        for it in range(1, iteration + 1):
            # Distance matrix for all xi points
            self.distance_matrix = self.calculate_distance_matrix(self.x)
            print("{}. iteration is started...".format(it))

            x_stars = []
            mi_list = []
            for i in range(len(x)):
                mi = self.calculate_mi(i, self.distance_matrix[i])
                mi_list.append(mi)
                xn = self.calculate_xn(i, self.distance_matrix[i])

                x_impostor = self.calculate_impostor_mean(mi, self.distance_matrix[i], i)
                if np.isnan(x_impostor).any():
                    x_h = x[i].copy()
                else:
                    x_h = x_impostor + mi * (x[i] - x_impostor) / np.linalg.norm(x[i] - x_impostor)

                xi_new = (1 - self.a) * xn + self.a * x_h
                x_stars.append(xi_new)

            mi_mean = np.array(mi_list).mean()
            self.mean_mi.append(mi_mean)
            print(f'Iteration {it}\nAvg margin:', mi_mean)
            self.x = np.array(x_stars)

            # leave-one-out cross validation with knn classifier
            acc = self.leave_one_out_cross_validation()
            print(f'Accuracy:', acc)
            self.acc.append(acc)


            # plot_data_3d(self.x, self.y, title=f'Iteration {it}')
            # plot_data_2d(self.x, self.y, title=f'Iteration {it}')

        return self.x

    def leave_one_out_cross_validation(self):
        scores = []
        fold = self.kfold.split(self.x, self.y)
        for k, (train, test) in enumerate(fold):
            knn = KNeighborsClassifier(n_neighbors=self.k)
            knn.fit(self.x[train], self.y[train])
            y_pred = knn.predict(self.x[test])
            scores.append(np.sum(self.y[test] == y_pred) / len(self.y[test]))
        return np.array(scores).mean()

    def drag_t(self, i, j):
        vector = self.distance_matrix[i]
        vector[i] = np.inf
        vector = np.where(self.y == self.y[i], vector, np.inf)
        k_nearest = np.argsort(vector)[:self.k]
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
        k_neighbors = np.argsort(vector)[-self.k:]
        result = self.x[k_neighbors]
        return result

    def plot_accuracy(self):
        plt.plot(list(range(1, len(self.acc) + 1)), self.acc)
        plt.title('Accuracy Score by Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.show()

    def plot_mean_mi(self):
        plt.plot(list(range(1, len(self.mean_mi) + 1)), self.mean_mi)
        plt.title('Mean Mi Score by Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Mi')
        plt.show()

