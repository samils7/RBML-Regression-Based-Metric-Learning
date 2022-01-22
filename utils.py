import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


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


def plot_accuracy(acc):
    if isinstance(acc[0], tuple):
        acc = np.array(acc)[:, 0]
    plt.plot(list(range(1, len(acc) + 1)), acc)
    plt.ylim(0.9, 1)
    plt.title('Accuracy Scores')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()


def plot_mean_mi(mean_mi_list):
    plt.plot(list(range(1, len(mean_mi_list) + 1)), mean_mi_list)
    plt.title('Average Margins')
    plt.xlabel('Iteration')
    plt.ylabel('Average Margin')
    plt.show()


def evaluate(x, y, dataset):
    if dataset in ['iris', 'wine', 'sonar']:
        acc = k_fold_cross_validation(x, y, k=len(x))
        return acc
    elif dataset in ['vowel', 'balance', 'pima']:

        """
        250 samples were randomly selected as a training set and the rest were used to define the test set. 
        Hence, 278, 375, and 518 test samples were available for each dataset, respectively. 
        This process was repeated 10 times independently. 
        For each dataset and each method, the average accuracy 
        and the corresponding standard deviation values were computed.
        """
        acc = k_fold_cross_validation(x, y, k=4)
        return acc

    elif dataset in ['segmentation', 'letters']:
        accs = []
        for i in range(10):
            acc = k_fold_cross_validation(x, y, k=10)
            accs.append(acc)
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        return avg_acc, std_acc


def k_fold_cross_validation(x, y, k=10, k_neighbors=3):
    # n_splits = len(dataset), this is equivalent to the Leave One Out strategy,
    kfold = KFold(n_splits=k)
    fold = kfold.split(x, y)
    scores = []
    for train, test in fold:
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        knn.fit(x[train], y[train])
        y_pred = knn.predict(x[test])
        scores.append(np.sum(y[test] == y_pred) / len(y[test]))
    return np.array(scores).mean()