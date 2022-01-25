import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from rbml import RBML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris, load_wine
from metric_learn import LMNN
import pandas as pd
import utils


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset name')
    parser.add_argument('--a', type=float, default=0.2,
                        help='alpha RBML')
    parser.add_argument('--b', type=float, default=2,
                        help='beta RBML')
    parser.add_argument('--k_neighbors', type=int, default=3,
                        help='k_neighbors RBML')
    parser.add_argument('--iteration', type=int, default=5,
                        help='iteration RBML')
    return parser.parse_args()


def zscore_normalization(dataset_x, dataset_y):
    scaler = StandardScaler()
    print(f'{dataset_name} dataset z-score normalization...')
    return scaler.fit_transform(dataset_x), dataset_y


class Pipeline:
    def __init__(self, dataset_name, a=0.5, b=2, k_neighbors=3):
        self.scaler_lmnn = StandardScaler()
        self.scaler_rbml = StandardScaler()
        self.scaler_rf = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        self.knn_lmnn = KNeighborsClassifier(n_neighbors=k_neighbors)
        self.knn_raw = KNeighborsClassifier(n_neighbors=k_neighbors)
        self.lmnn = LMNN(k=k_neighbors, verbose=False)
        self.rbml = RBML(a=a, b=b, k_neighbors=k_neighbors, dataset=dataset_name)
        self.random_forest = None
        self.rf_projected_target = None

    def fit(self, train_x, train_y, iteration=4):
        self.knn_raw.fit(train_x, train_y)
        self.lmnn.fit(train_x, train_y)
        lmnn_projected = self.lmnn.transform(train_x)
        self.knn_lmnn.fit(lmnn_projected, train_y)
        lmnn_projected = self.scaler_lmnn.fit_transform(lmnn_projected)
        rbml_projected = self.rbml.fit_transform(x=lmnn_projected, y=train_y, iteration=iteration)
        #print(' --> '.join([str(np.round(m, 2)) for m in self.rbml.avg_margins]))
        rbml_projected = self.scaler_rbml.fit_transform(rbml_projected)
        self.random_forest = RandomForestRegressor(n_estimators=train_x.shape[1])
        self.random_forest.fit(lmnn_projected, rbml_projected)
        rf_projected = self.random_forest.predict(lmnn_projected)
        self.rf_projected_target = rf_projected
        rf_projected = self.scaler_rf.fit_transform(rf_projected)
        self.knn.fit(rf_projected, train_y)

    def transform(self, test_x):
        lmnn_projected = self.lmnn.transform(test_x)
        lmnn_projected = self.scaler_lmnn.transform(lmnn_projected)
        rf_projected = self.random_forest.predict(lmnn_projected)
        rf_projected = self.scaler_rf.transform(rf_projected)
        return self.knn.predict(rf_projected)

    def score(self, test_x, test_y):
        result = self.transform(test_x)
        return accuracy_score(test_y, result)

    def lmnn_score(self, test_x, test_y):
        lmnn_projected = self.lmnn.transform(test_x)
        result = self.knn_lmnn.predict(lmnn_projected)
        return accuracy_score(test_y, result)

    def raw_score(self, test_x, test_y):
        result = self.knn_raw.predict(test_x)
        return accuracy_score(test_y, result)


def process_pipeline_1(dataset_x, dataset_y, dataset_name, a=0.5, b=2, k_neighbors=3, iteration=4):
    dataset_x, dataset_y = zscore_normalization(dataset_x, dataset_y)
    kfold = KFold(n_splits=len(dataset_x), shuffle=True, random_state=42)
    fold = kfold.split(dataset_x, dataset_y)
    raw_scores = []
    lmnn_scores = []
    rbml_scores = []
    pipeline = None
    for train, test in fold:
        pipeline = Pipeline(dataset_name, a=a, b=b, k_neighbors=k_neighbors)
        pipeline.fit(dataset_x[train], dataset_y[train], iteration=iteration)
        raw_scores.append(pipeline.raw_score(dataset_x[test], dataset_y[test]))
        lmnn_scores.append(pipeline.lmnn_score(dataset_x[test], dataset_y[test]))
        rbml_scores.append(pipeline.score(dataset_x[test], dataset_y[test]))

    avg_margins = pipeline.rbml.avg_margins
    utils.plot_mean_mi(avg_margins, save_path=f'{dataset_name}_mean_mi.png',
                       title=f'{dataset_name} Dataset Average Margins')
    utils.plot_dataset(dataset_x, dataset_y, save_path=f'{dataset_name}_raw_dataset.png',
                       title=f'{dataset_name} Raw Dataset')
    utils.plot_dataset(pipeline.rbml.x, pipeline.rbml.y, save_path=f'{dataset_name}_rbml_trained.png',
                       title=f'{dataset_name} Dataset RBML Projected')
    utils.plot_dataset(pipeline.rf_projected_target, pipeline.rbml.y,
                       save_path=f'{dataset_name}_rf_projected_target.png',
                       title=f'{dataset_name} Dataset RF Projected Target')

    print(f'Euclidean Accuracy: {np.mean(raw_scores):.3f}')
    print(f'LMNN Accuracy: {np.mean(lmnn_scores):.3f}')
    print(f'RBML Accuracy: {np.mean(rbml_scores):.3f}')


def process_pipeline_2(dataset_x, dataset_y, dataset_name, a=0.5, b=2, k_neighbors=3, iteration=4):
    """For the Vowel, Balance and Pima datasets, 250 samples were randomly selected as a training set and the rest were used to define the test set.
    Hence, 278, 375, and 518 test samples were available for each dataset, respectively.
    This process was repeated 10 times independently.
    For each dataset and each method, the average accuracy and the corresponding standard deviation values were computed."""
    dataset_x, dataset_y = zscore_normalization(dataset_x, dataset_y)
    raw_scores = []
    lmnn_scores = []
    rbml_scores = []
    pipeline = None
    for p in range(10):
        # shuffle dataset
        shuffle_index = np.random.permutation(len(dataset_x))
        dataset_x, dataset_y = dataset_x[shuffle_index], dataset_y[shuffle_index]
        # split dataset
        train_x, test_x, train_y, test_y = dataset_x[:250], dataset_x[250:], dataset_y[:250], dataset_y[250:]

        pipeline = Pipeline(dataset_name, a=a, b=b, k_neighbors=k_neighbors)
        pipeline.fit(train_x, train_y, iteration=iteration)
        raw_scores.append(pipeline.raw_score(test_x, test_y))
        lmnn_scores.append(pipeline.lmnn_score(test_x, test_y))
        rbml_scores.append(pipeline.score(test_x, test_y))

    avg_margins = pipeline.rbml.avg_margins
    utils.plot_mean_mi(avg_margins, save_path=f'{dataset_name}_mean_mi.png',
                       title=f'{dataset_name} Dataset Average Margins')
    utils.plot_dataset(dataset_x, dataset_y, save_path=f'{dataset_name}_raw_dataset.png',
                       title=f'{dataset_name} Raw Dataset')
    utils.plot_dataset(pipeline.rbml.x, pipeline.rbml.y, save_path=f'{dataset_name}_rbml_trained.png',
                       title=f'{dataset_name} Dataset RBML Projected')
    utils.plot_dataset(pipeline.rf_projected_target, pipeline.rbml.y,
                       save_path=f'{dataset_name}_rf_projected_target.png',
                       title=f'{dataset_name} Dataset RF Projected Target')

    print(f'Euclidean Accuracy: {np.mean(raw_scores):.3f}, ({np.std(raw_scores):.3f})')
    print(f'LMNN Accuracy: {np.mean(lmnn_scores):.3f}, ({np.std(lmnn_scores):.3f})')
    print(f'RBML Accuracy: {np.mean(rbml_scores):.3f}, ({np.std(rbml_scores):.3f})')


def process_pipeline_3(dataset_x, dataset_y, dataset_name, a=0.5, b=2, k_neighbors=3, iteration=4):
    dataset_x, dataset_y = zscore_normalization(dataset_x, dataset_y)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = kfold.split(dataset_x, dataset_y)
    raw_scores = []
    lmnn_scores = []
    rbml_scores = []
    pipeline = None
    for train, test in fold:
        pipeline = Pipeline(dataset_name, a=a, b=b, k_neighbors=k_neighbors)
        pipeline.fit(dataset_x[train], dataset_y[train], iteration=iteration)
        raw_scores.append(pipeline.raw_score(dataset_x[test], dataset_y[test]))
        lmnn_scores.append(pipeline.lmnn_score(dataset_x[test], dataset_y[test]))
        rbml_scores.append(pipeline.score(dataset_x[test], dataset_y[test]))

    avg_margins = pipeline.rbml.avg_margins
    utils.plot_mean_mi(avg_margins, save_path=f'{dataset_name}_mean_mi.png',
                       title=f'{dataset_name} Dataset Average Margins')
    utils.plot_dataset(dataset_x, dataset_y, save_path=f'{dataset_name}_raw_dataset.png',
                       title=f'{dataset_name} Raw Dataset')
    utils.plot_dataset(pipeline.rbml.x, pipeline.rbml.y, save_path=f'{dataset_name}_rbml_trained.png',
                       title=f'{dataset_name} Dataset RBML Projected')
    utils.plot_dataset(pipeline.rf_projected_target, pipeline.rbml.y,
                       save_path=f'{dataset_name}_rf_projected_target.png',
                       title=f'{dataset_name} Dataset RF Projected Target')

    print(f'Euclidean Accuracy: {np.mean(raw_scores):.3f}, ({np.std(raw_scores):.3f})')
    print(f'LMNN Accuracy: {np.mean(lmnn_scores):.3f}, ({np.std(lmnn_scores):.3f})')
    print(f'RBML Accuracy: {np.mean(rbml_scores):.3f}, ({np.std(rbml_scores):.3f})')


if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'all':
        #dataset_names = reversed(['iris', 'wine', 'sonar','vowel', 'balance', 'pima', 'segmentation', 'letters'])
        # dataset_names = ['iris', 'wine', 'sonar']
        #dataset_names = ['vowel', 'balance', 'pima']
        dataset_names = ['segmentation', 'letters']
    elif args.dataset == 'group1':
        dataset_names = ['iris', 'wine', 'sonar']
    elif args.dataset == 'group2':
        dataset_names = ['vowel', 'balance', 'pima']
    elif args.dataset == 'group3':
        dataset_names = ['segmentation', 'letters']
    else:
        dataset_names = [args.dataset]

    alpha, beta, k_neighbors, iteration = args.a, args.b, args.k_neighbors, args.iteration
    for dataset_name in dataset_names:
        print(f'Processing {dataset_name} dataset...')
        if dataset_name == 'iris':
            iris = load_iris()
            dataset_x, dataset_y = iris.data, iris.target
            #alpha, beta, k_neighbors, iteration = 0.2, 2, 3, 5
            process_pipeline_1(dataset_x, dataset_y, dataset_name, a=alpha, b=beta, k_neighbors=k_neighbors, iteration=iteration)

        elif dataset_name == 'wine':
            wine = load_wine()
            dataset_x, dataset_y = wine.data, wine.target
            #alpha, beta, k_neighbors, iteration = 0.2, 2, 3, 5
            process_pipeline_1(dataset_x, dataset_y, dataset_name, a=alpha, b=beta, k_neighbors=k_neighbors, iteration=iteration)

        elif dataset_name == 'sonar':
            dataset = pd.read_csv('datasets/sonar/sonar.all-data', header=None)
            array = dataset.values
            dataset_x = array[:, :-1].astype(float)
            dataset_y = array[:, -1]
            dataset_y[dataset_y == 'R'] = 0
            dataset_y[dataset_y == 'M'] = 1
            dataset_y = dataset_y.astype(int)
            #alpha, beta, k_neighbors, iteration = 0.2, 2, 3, 4
            process_pipeline_1(dataset_x, dataset_y, dataset_name, a=alpha, b=beta, k_neighbors=k_neighbors, iteration=iteration)

        elif dataset_name == 'vowel':
            #dataset = pd.read_csv("datasets/vowel/vowel-context.data", delimiter="\s+", header=None)
            dataset = pd.read_csv("datasets/vowel/vowel.tr-orig-order", header=None)
            array = dataset.values
            dataset_x = array[:, :-1].astype(float)
            dataset_y = array[:, -1]
            #alpha, beta, k_neighbors, iteration = 0.2, 2, 3, 4
            process_pipeline_2(dataset_x, dataset_y, dataset_name, a=alpha, b=beta, k_neighbors=k_neighbors, iteration=iteration)

        elif dataset_name == 'balance':
            dataset = pd.read_csv("datasets/balance_scale/balance-scale.data", delimiter=",", header=None)
            array = dataset.values
            dataset_x = array[:, 1:].astype(float)
            dataset_y = array[:, 0]
            dataset_y[dataset_y == 'B'] = 0
            dataset_y[dataset_y == 'R'] = 1
            dataset_y[dataset_y == 'L'] = 2
            dataset_y = dataset_y.astype(int)
            #alpha, beta, k_neighbors, iteration = 0.2, 2, 3, 4
            process_pipeline_2(dataset_x, dataset_y, dataset_name, a=alpha, b=beta, k_neighbors=k_neighbors, iteration=iteration)

        elif dataset_name in ['pima', 'diabetes']:
            pima = pd.read_csv("datasets/pima-indians-diabetes.csv", delimiter=",", header=None)
            dataset_x, dataset_y = pima.iloc[:, :-1].values, pima.iloc[:, -1].values
            #alpha, beta, k_neighbors, iteration = 0.2, 2, 3, 4
            process_pipeline_2(dataset_x, dataset_y, dataset_name, a=alpha, b=beta, k_neighbors=k_neighbors, iteration=iteration)

        elif dataset_name == 'segmentation':
            dataset1 = pd.read_csv("datasets/image_segmentation/segmentation.data", delimiter=",", header=None)
            dataset2 = pd.read_csv("datasets/image_segmentation/segmentation.test", delimiter=",", header=None)
            concat_dataset = pd.concat([dataset1, dataset2])
            dataset_x = concat_dataset.iloc[:, 1:].values
            dataset_y = concat_dataset.iloc[:, 0].values
            label_encoder = LabelEncoder()
            dataset_y = label_encoder.fit_transform(dataset_y)
            #alpha, beta, k_neighbors, iteration = 0.2, 2, 3, 4
            process_pipeline_3(dataset_x, dataset_y, dataset_name, a=alpha, b=beta, k_neighbors=k_neighbors, iteration=iteration)

        elif dataset_name == 'letters':
            dataset = pd.read_csv("datasets/letter/letter-recognition.data", delimiter=",", header=None)
            dataset_x = dataset.iloc[:, 1:].values
            dataset_y = dataset.iloc[:, 0].values
            label_encoder = LabelEncoder()
            dataset_y = label_encoder.fit_transform(dataset_y)
            #alpha, beta, k_neighbors, iteration = 0.2, 2, 3, 4
            process_pipeline_3(dataset_x, dataset_y, dataset_name, a=alpha, b=beta, k_neighbors=k_neighbors, iteration=iteration)

        else:
            raise ValueError('dataset_name must be one of iris, pima/diabetes, wine, sonar')
