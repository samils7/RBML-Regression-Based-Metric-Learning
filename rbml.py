from sklearn.datasets import load_iris
import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class RBML:
    def __init__(self, k=3, b=3, a=0.3):
        self.b = b
        self.a = a
        self.k = k
        self.x = None
        self.y = None
        self.distance_matrix = None
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.loo = LeaveOneOut()
        self.kfold = None

    def fit(self, x, y, iteration=1):
        self.x = x
        self.y = y
        self.kfold = KFold(n_splits=len(self.x))

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

            print(f'Iteration {it}\nAvg margin:', np.array(mi_list).mean())
            self.x = np.array(x_stars)

            # leave-one-out cross validation with knn classifier
            acc = self.leave_one_out_cross_validation()
            print(f'Accuracy:', acc)

            #plot_data_3d(self.x, self.y, title=f'Iteration {it}')
            #plot_data_2d(self.x, self.y, title=f'Iteration {it}')

        return self.x

    def leave_one_out_cross_validation(self):
        scores = []
        fold = self.kfold.split(self.x, self.y)
        for k, (train, test) in enumerate(fold):
            self.knn.fit(self.x[train], self.y[train])
            y_pred = self.knn.predict(self.x[test])
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


def plot_data_3d(x, y, save_path=None, title=None):
    x = PCA(n_components=3).fit_transform(x)
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_data_2d(x, y, save_path=None, title=None):
    x = PCA(n_components=2).fit_transform(x)
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    ax = fig.add_subplot(111)
    ax.scatter(x[:, 0], x[:, 1], c=y)
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    iris = load_iris()
    X = iris['data']
    y = iris['target']

    #plot_data_3d(X, y, save_path='iris_3d.png', title='Iris dataset projected to 3D')
    plot_data_2d(X, y, save_path='iris_2d.png', title='Iris dataset projected to 2D')

    rbml = RBML()
    x_rbml = rbml.fit(x=X, y=y, iteration=5)

    #plot_data_3d(rbml.x, y, save_path='iris_3d_rbml.png', title='Iris dataset projected to 3D with RBML')
    plot_data_2d(rbml.x, y, save_path='iris_2d_rbml.png', title='Iris dataset projected to 2D with RBML')

    # random forest regression to learn regression from iris dataset to x_transformed
    x_train, x_test, rbml_train, rbml_test = train_test_split(X, x_rbml, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=4, random_state=42)
    rf.fit(x_train, rbml_train)
    x_transformed = rf.predict(x_test)
    print('regression score:', rf.score(x_test, rbml_test))

    # project all data with learned regression and plot 3D
    iris_projected = rf.predict(X)
    #plot_data_3d(iris_projected, y, save_path='iris_3d_transformed.png', title='Iris dataset projected to 3D with Random Forest')
    plot_data_2d(iris_projected, y, save_path='iris_2d_transformed.png', title='Iris dataset projected to 2D with Random Forest')
