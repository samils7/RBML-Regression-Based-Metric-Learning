from sklearn.neighbors import KNeighborsClassifier

from rbml import RBML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes, load_wine
from metric_learn import LMNN
import pandas as pd
import utils


def main(dataset_x, dataset_y, dataset_name, alpha=0.5, beta=2, k_neighbors=3, iteration=4):
    # z score
    scaler = StandardScaler()
    dataset_x = scaler.fit_transform(dataset_x)
    # utils.plot_data_3d(dataset_x, dataset_y, save_path=f'{dataset_name}_3d.png', title=f'{dataset_name} dataset visualization on 3D')
    utils.plot_data_2d(dataset_x, dataset_y, save_path=f'{dataset_name}_2d.png', title=f'{dataset_name} dataset visualization on 2D')

    initial_eval = utils.evaluate(dataset_x, dataset_y, dataset_name)
    print(f'Initial Evaluation Score:{initial_eval:.3f}')

    # LMNN projection
    lmnn_x_train, lmnn_x_test, lmnn_y_train, lmnn_y_test = train_test_split(dataset_x, dataset_y, test_size=0.7, random_state=42)
    lmnn = LMNN(k=k_neighbors, random_state=None, verbose=False)
    lmnn.fit(lmnn_x_train, lmnn_y_train)
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(lmnn.transform(lmnn_x_train), lmnn_y_train)
    lmnn_y_pred = knn.predict(lmnn.transform(lmnn_x_test))
    lmnn_acc = knn.score(lmnn.transform(lmnn_x_test), lmnn_y_test)
    #lmnn_acc = utils.evaluate(lmnn_projected, dataset_y, dataset_name)
    print(f'LMNN Evaluation Score: {lmnn_acc:.3f}')

    # RBML projection
    rbml = RBML(alpha, beta, k_neighbors, dataset=dataset_name)
    x_rbml = rbml.fit_transform(x=dataset_x, y=dataset_y, iteration=iteration)
    rbml_acc = utils.evaluate(x_rbml, dataset_y, dataset_name)
    print(f'RBML Evaluation Score: {rbml_acc:.3f}')

    #utils.plot_accuracy(rbml.evaluation_scores)
    utils.plot_mean_mi(rbml.avg_margins)
    # utils.plot_data_3d(rbml.x, dataset_y, save_path=f'{dataset_name}_3d_rbml.png', title=f'{dataset_name} dataset after RBML\nRBML Evaluation Score: {rbml_acc:.3f}')
    utils.plot_data_2d(rbml.x, dataset_y, save_path=f'{dataset_name}_2d_rbml.png', title=f'{dataset_name} dataset after RBML\nRBML Evaluation Score: {rbml_acc:.3f}')

    # random forest regression to learn regression from dataset to x_transformed
    x_train, x_test, rbml_train, rbml_test = train_test_split(dataset_x, x_rbml, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=x_train.shape[1], random_state=42)
    rf.fit(x_train, rbml_train)
    print(f'Projection Function Regression Score: {rf.score(x_test, rbml_test):.3f}')

    # project all data with learned regression and plot 3D
    dataset_projected = rf.predict(dataset_x)
    final_acc = utils.evaluate(dataset_projected, dataset_y, dataset_name)
    print(f'Final Evaluation Score: {final_acc:.3f}')
    # utils.plot_data_3d(dataset_projected, dataset_y, save_path=f'{dataset_name}_3d_transformed.png', title=f'{dataset_name} dataset after RBML + Random Forest Regression\nFinal Evaluation Score: {final_acc:.3f}')
    utils.plot_data_2d(dataset_projected, dataset_y, save_path=f'{dataset_name}_2d_transformed.png', title=f'{dataset_name} dataset after RBML + Random Forest Regression\nFinal Evaluation Score: {final_acc:.3f}')


if __name__ == '__main__':
    dataset_name = 'sonar'
    if dataset_name == 'iris':
        iris = load_iris()
        dataset_x, dataset_y = iris.data, iris.target
    elif dataset_name == 'wine':
        wine = load_wine()
        dataset_x, dataset_y = wine.data, wine.target
    elif dataset_name == 'sonar':
        dataset = pd.read_csv('datasets/sonar/sonar.all-data', header=None)
        array = dataset.values
        dataset_x = array[:, :-1].astype(float)
        dataset_y = array[:, -1]
        dataset_y[dataset_y == 'R'] = 0
        dataset_y[dataset_y == 'M'] = 1
        dataset_y = dataset_y.astype(int)

    elif dataset_name == 'vowel':
        pass
    elif dataset_name == 'balance':
        pass
    elif dataset_name in ['pima', 'diabetes']:
        diabetes = load_diabetes()
        dataset_x, dataset_y = diabetes.data, diabetes.target

    elif dataset_name == 'segmentation':
        pass
    elif dataset_name == 'letters':
        pass

    else:
        raise ValueError('dataset_name must be one of iris, pima/diabetes, wine, sonar')

    main(dataset_x, dataset_y, 'wine', alpha=0.2, beta=2, iteration=10)
