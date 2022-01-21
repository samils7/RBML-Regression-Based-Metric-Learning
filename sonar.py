from rbml import RBML
import visualUtil
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    dataset = pd.read_csv('datasets/sonar.csv', header=None)
    array = dataset.values
    X = array[:, 0:-1].astype(float)
    y = array[:, -1]

    # z score
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # plot_data_3d(X, y, save_path='wine_3d.png', title='Wine dataset projected to 3D')
    visualUtil.plot_data_2d(X, y, save_path='wine_2d.png', title='Wine dataset projected to 2D')

    rbml = RBML(b=2, a=0.3)
    x_rbml = rbml.fit(x=X, y=y, iteration=50)
    rbml.plot_accuracy()
    rbml.plot_mean_mi()

    # plot_data_3d(rbml.x, y, save_path='wine_3d_rbml.png', title='Wine dataset projected to 3D with RBML')
    visualUtil.plot_data_2d(rbml.x, y, save_path='wine_2d_rbml.png', title='Wine dataset projected to 2D with RBML')

    # random forest regression to learn regression from wine dataset to x_transformed
    x_train, x_test, rbml_train, rbml_test = train_test_split(X, x_rbml, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=4, random_state=42)
    rf.fit(x_train, rbml_train)
    x_transformed = rf.predict(x_test)
    print('regression score:', rf.score(x_test, rbml_test))

    # project all data with learned regression and plot 3D
    wine_projected = rf.predict(X)
    # plot_data_3d(wine_projected, y, save_path='wine_3d_transformed.png', title='Wine dataset projected to 3D with Random Forest')
    visualUtil.plot_data_2d(wine_projected, y, save_path='wine_2d_transformed.png',
                 title='Wine dataset projected to 2D with Random Forest')

