import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
