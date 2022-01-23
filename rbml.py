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
        self.evaluation_scores = []

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

            # leave-one-out cross validation with knn classifier
            eval_score_acc, eval_score_std = self.evaluate()
            self.evaluation_scores.append(eval_score_acc)

            print(f'Iteration {it}\t\tAvg margin: {self.avg_margins[-1]:.3f}\tEvaluation score: {eval_score_acc:.3f}')
            # plot_data_3d(self.x, self.y, title=f'Iteration {it}')
            # plot_data_2d(self.x, self.y, title=f'Iteration {it}')

        return self.x

    def evaluate(self):
        if self.dataset in ['iris', 'wine', 'sonar']:
            acc, std = self.k_fold_cross_validation(k=len(self.x))
            return acc, std
        elif self.dataset in ['vowel', 'balance', 'pima']:
            pass
            """
            250 samples were randomly selected as a training set and the rest were used to define the test set. 
            Hence, 278, 375, and 518 test samples were available for each dataset, respectively. 
            This process was repeated 10 times independently. 
            For each dataset and each method, the average accuracy 
            and the corresponding standard deviation values were computed.
            """
            acc, std = self.k_fold_cross_validation(k=10)

            return acc, std

        elif self.dataset in ['segmentation', 'letters']:
            acc, std = self.k_fold_cross_validation(k=10)
            return acc, std

    def k_fold_cross_validation(self, k=10):
        # n_splits = len(dataset), this is equivalent to the Leave One Out strategy,
        kfold = KFold(n_splits=k)
        fold = kfold.split(self.x, self.y)
        scores = []
        for train, test in fold:
            knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)
            knn.fit(self.x[train], self.y[train])
            y_pred = knn.predict(self.x[test])
            scores.append(np.sum(self.y[test] == y_pred) / len(self.y[test]))
        return np.array(scores).mean(), np.array(scores).std()

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
