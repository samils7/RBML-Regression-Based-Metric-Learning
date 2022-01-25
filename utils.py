import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


def plot_dataset(x, y, save_path=None, title=None):
    plot_data_3d(x, y, save_path='3d_'+save_path, title=title)
    plot_data_2d(x, y, save_path='2d_'+save_path, title=title)


def plot_data_3d(x, y, save_path=None, title=None):
    x = PCA(n_components=3).fit_transform(x)
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
    if save_path:
        plt.savefig('plots/'+save_path)
    plt.show()


def plot_data_2d(x, y, save_path=None, title=None):
    x = PCA(n_components=2).fit_transform(x)
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    ax = fig.add_subplot(111)
    ax.scatter(x[:, 0], x[:, 1], c=y)
    if save_path:
        plt.savefig('plots/'+save_path)
    plt.show()


def plot_accuracy(acc, save_path=None):
    if isinstance(acc[0], tuple):
        acc = np.array(acc)[:, 0]
    plt.plot(list(range(1, len(acc) + 1)), acc)
    plt.ylim(0.9, 1)
    plt.title('Accuracy Scores')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    if save_path:
        plt.savefig('plots/'+save_path)
    plt.show()


def plot_mean_mi(mean_mi_list, save_path=None, title='Average Margin'):
    plt.plot(np.arange(1, len(mean_mi_list) + 1), mean_mi_list)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Average Margin')
    if save_path:
        plt.savefig('plots/'+save_path)
    plt.show()
