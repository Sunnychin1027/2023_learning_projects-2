"""
File: boston_housing_competition.py
Name: Sunny
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""
import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection, preprocessing, metrics, decomposition, tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample
import math

TEST_FILE = 'boston_housing/test.csv'
TRAIN_FILE = 'boston_housing/train.csv'


def main():
    # Load the Boston housing dataset
    x = pd.read_csv(TRAIN_FILE)
    x.pop('ID')

    # Extract true labels
    y = x['medv']
    x.pop('medv')
    # print('true labe', y)
    # print('data set', x)

    # test_data to do predictions
    test_data = pd.read_csv(TEST_FILE)
    test_data.pop('ID')

    # Data preprocessing (check missing value, scaling data, splitting dataset)
    standardizer = preprocessing.StandardScaler()
    x = standardizer.fit_transform(x)
    test_data = standardizer.transform(test_data)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)
    # print(x_train.head())
    # print(x_val.head())
    # print(y_train.head())
    # print(y_val.head())

    ################
    # Gradient Boosting Regressor
    # gbc predict1 rmse:  0.9614239990962217
    # gbc predict2 rmse:  3.3355244662393093
    # outfile = gbc2.csv

    gbr = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, min_samples_leaf=3, min_samples_split=4)
    gbc = gbr.fit(x_train, y_train)
    predict1 = gbc.predict(x_train)
    predict2 = gbc.predict(x_val)
    print('gbc predict1 rmse: ', metrics.mean_squared_error(predict1, y_train)**0.5)
    print('gbc predict2 rmse: ', metrics.mean_squared_error(predict2, y_val)**0.5)
    # predictions = gbc.predict(test_data)
    # out_file(predictions, 'gbc2.csv')

    # Do grid search
    # Best hyper_parameters:  {'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 3, 'min_samples_split': 4}
    # Best score:  0.8156269549044873
    param_grid = {'learning_rate': [0.1, 0.02, 0.05], 'max_depth': [2, 3, 4], 'min_samples_leaf': [3,5], 'min_samples_split':[2,3,4]}
    gb = GradientBoostingRegressor()
    grid_search = GridSearchCV(gb, param_grid=param_grid, cv=5)  # cv = number of folds in cross validation
    print('--finding--')
    grid_search.fit(x_train, y_train)
    print('Best hyper parameters: ', grid_search.best_params_)
    print('Best score: ', grid_search.best_score_)
    ################

    ################
    # Support Vector Regression
    # ==== 1. svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.1) ====
    # predict1 rmse:  0.5003054695376171
    # predict2 rmse:  0.4526105558975654
    # outfile = SVR3.csv
    # ==== 2. svr_rbf = SVR(kernel='rbf', C=10000, gamma=0.1) ====
    # predict1 rmse:  0.09929872221780345
    # predict2 rmse:  0.09961177326383881
    # outfile = SVR4.csv

    # svr_rbf = SVR(kernel='rbf', C=10000, gamma=0.1)
    # svr_rbf_classifier = svr_rbf.fit(x_train, y_train)
    # predict1 = svr_rbf_classifier.predict(x_train)
    # print('predict1 rmse: ', metrics.mean_squared_error(predict1, y_train)**0.5)
    # svr_rbf_classifier = svr_rbf.fit(x_val, y_val)
    # predict2 = svr_rbf_classifier.predict(x_val)
    # print('predict2 rmse: ', metrics.mean_squared_error(predict2, y_val) ** 0.5)
    # predictions = svr_rbf_classifier.predict(test_data)
    # out_file(predictions, 'SVR4.csv')

    # Do grid search
    # Best hyper_parameters:  {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}
    # Best score:  0.8148787787591802
    # param_grid = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['rbf', 'linear', 'poly'], 'gamma': [0.1, 0.01, 0.001]}
    # svr = SVR()
    # grid_search = GridSearchCV(svr, param_grid=param_grid, cv=5)  # cv = number of folds in cross validation
    # print('--finding--')
    # grid_search.fit(x_val, y_val)
    # print('Best hyper parameters: ', grid_search.best_params_)
    # print('Best score: ', grid_search.best_score_)

    # Do grid search with bootstrapping
    # n_iterations = 10
    # boot_strap_size = len(x_train)
    # svr = SVR()
    # param_grid = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['rbf', 'linear', 'poly'], 'gamma': [0.1, 0.01, 0.001]}
    # best_scores = []
    # best_params = []
    # for _ in range(n_iterations):
    #     # Create bootstrap sample
    #     x_train_bs, y_train_bs = resample(x_train, y_train, n_samples=boot_strap_size, random_state=42, replace=True)
    #     # Perform grid search with cross validation
    #     grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
    #     grid_search.fit(x_train_bs, y_train_bs)
    #     best_score = grid_search.best_score_
    #     best_param = grid_search.best_params_
    #     best_scores.append(best_score)
    #     best_params.append(best_param)
    # avg_score = sum(best_scores)/n_iterations
    # best_parameters = max(set(best_param), key=best_params.count)
    # print('avg_score: ', avg_score, 'best_parameters: ', best_parameters)
    ################

    ################
    # Random forest
    # random forest regressor predict1:  1.5464764332153806
    # random forest regressor predict2:  3.248523636241144
    # 100 times avg train:  1.4063756908312763
    # 100 times avg valid:  3.2538932460776073
    # total_err = 0
    # for i in range(100):
    #     x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)
    #     forest = ensemble.RandomForestRegressor()
    #     forest_classifier = forest.fit(x_train, y_train)
    #     predict1 = forest_classifier.predict(x_train)
    #     predict2 = forest_classifier.predict(x_val)
    #     val_err = metrics.mean_squared_error(predict1, y_train)**0.5
    #     total_err += val_err
    # print('100 times avg train: ', total_err/100)
    # print('random forest regressor predict1: ', metrics.mean_squared_error(predict1, y_train)**0.5)
    # print('random forest regressor predict2: ', metrics.mean_squared_error(predict2, y_val)**0.5)
    # x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)
    # forest = ensemble.RandomForestRegressor()
    # forest_classifier = forest.fit(x_train, y_train)
    # predictions = forest_classifier.predict(test_data)
    # out_file(predictions, 'random_forest1')

    ################

    ################
    # Construct Tree ==>> Find the likely important factor ==>> rm, lstat, dis, rad, ptratio
    # 100 times d_tree val error avg:  4.590747093384838
    # x_train RMSE:  4.259764631069938
    # x_val RMSE:  4.590747093384846
    # 100 times val error avg:  4.590747093384838
    # Outfile: d_tree.csv
    # features_name = ['crim', 'zn', 'indus', 'chas','nox', 'rm','age', 'dis', 'rad','tax', 'ptratio', 'black', 'lstat']
    # total_err = 0
    # for i in range(100):
    #     x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)
    #     d_tree = tree.DecisionTreeRegressor(max_depth=3, min_samples_leaf=3)
    #     d_tree_classifier = d_tree.fit(x_train, y_train)
    #     predictions = d_tree_classifier.predict(x_train)
    #     # print('x_train RMSE: ', metrics.mean_squared_error(predictions, y_train)**0.5)
    #     predictions2 = d_tree_classifier.predict(x_val)
    #     # print('x_val RMSE: ', metrics.mean_squared_error(predictions2, y_val)**0.5)
    #     val_err = metrics.mean_squared_error(predictions2, y_val) ** 0.5
    #     total_err += val_err
    # print('100 times d_tree val error avg: ', total_err / 100)
    # tree.export_graphviz(d_tree_classifier, out_file='train_tree', feature_names=features_name)
    # output file
    # predictions3 = d_tree_classifier.predict(test_data)
    # x_test = test_data[features_name]
    # out_file(predictions3, 'd_tree.csv')
    ################

    ################
    # Linear Regression
    # 100 times poly degree 2 val error avg:  9.01507966853467
    # b4 pca/ x train acc: 4.64497061561217
    # b4 pca/ x val acc: 5.416699125230135
    # after pca 6 x train acc: 5.6094376987082395
    # after pca 6 x val acc: 3.9533495348337486
    # poly2 x train acc: 4.2294013909842265
    # poly2 x val acc: 3.9639104663924054

    # pca = decomposition.PCA(n_components=6)
    # x_train = pca.fit_transform(x_train)
    # x_val = pca.transform(x_val)
    # test_data = pca.transform(test_data)
    # var_retained = sum(pca.explained_variance_ratio_)
    # print('var trained: ', var_retained)

    # total_err = 0
    # for i in range(100):
    #     x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)
    #     poly_fea_extractor = preprocessing.PolynomialFeatures(degree=2)
    #     x_train = poly_fea_extractor.fit_transform(x_train)
    #     x_val = poly_fea_extractor.transform(x_val)
    #     h = linear_model.LinearRegression()
    #     classifier = h.fit(x_train, y_train)
    #     predictions = classifier.predict(x_train)
    #     # print('x train acc: ', metrics.mean_squared_error(predictions, y_train)**0.5)
    #     predictions2 = classifier.predict(x_val)
    #     val_err = metrics.mean_squared_error(predictions2, y_val)**0.5
    #     total_err += val_err
    # print('100 times poly degree 2 val error avg: ', total_err/100)

    # print('x val acc: ', metrics.mean_squared_error(predictions2, y_val)**0.5)
    # test_data = poly_fea_extractor.transform(test_data)
    # test_prediction = classifier.predict(test_data)
    # Outfile
    # out_file(test_prediction, 'out_put_file2.csv')
    ################


def out_file(predictions, filename):
    print('\n===============================================')
    print(f'Writing predictions to --> {filename}')
    test_data = pd.read_csv(TEST_FILE)
    test_ids = test_data['ID']
    output_df = pd.DataFrame({'ID': test_ids, 'medv': predictions})
    output_df.to_csv(filename, index=False)
    print('\n===============================================')


if __name__ == '__main__':
    main()
