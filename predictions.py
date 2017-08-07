# -*- coding: utf-8 -*-
import math
from pprint import pprint
import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import svm
from matplotlib import pyplot as plt
import pickle
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneOut
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_rows', 1000)


def linear_regression(data):
    features = data.columns.tolist()
    features.remove('GID')
    features.remove('label')
    features.remove('look_times')
    # print features
    response = ['label']

    # declare a linear regression model 
    lr = LinearRegression(fit_intercept=True, normalize=True)
    # define response variable: label
    y = np.asarray(data[response])
    # define features
    X = data[features]

    # fit regression model to the data
    model = lr.fit(X, y)
    # 利用我們的model預測中選率
    predicted_y = model.predict(X) # predicted_y是一個雙層numpy array

# 有正規化w的線性回歸
def ridge_regression(data):
    features = data.columns.tolist()
    features.remove('label')
    response = ['label']
    # 宣告一個Ridge Regression model
    # lr = Ridge(normalize=True, alpha=0.01)
    lr = Ridge()
    # 定義應變數: label(需是一個DataFrame)
    y = data[response]
    # 定義features (需是一個DataFrame)
    X = data[features]
    

    leave_one_out(lr, X.values, y.values)
    
    # fit regression model to the data
    model = lr.fit(X, y)
    # 利用我們的model預測中選率
    predicted_y = model.predict(X) # predicted_y是一個雙層numpy array
    # 把原本y的DataFrame也轉成雙層numpy array，方便等等列印
    y = np.array(y)

    # 列印各個結果
    print_y_and_predicted_y(y, predicted_y)
    print_r2_score(y, predicted_y)
    print_coefficients(model, features)
    print_MSE(y, predicted_y)
    plot_true_and_pred_scatter(y, predicted_y)

def SVR(data):
    features = data.columns.tolist()
    features.remove('label')
    response = ['label']

    y = data[response]
    X = data[features]

    # if load_model:
    #     model = joblib.load('svr_rbf_kernel.pkl') 
    svr_algr = svm.SVR(C=1.0, kernel='rbf')
    # svr_algr = svm.SVR(C=1.0, kernel='linear')
    # fit regression model to the data
    model = svr_algr.fit(X, y.values.ravel())
    # joblib.dump(model, 'svr_linear_kernel.pkl')

    leave_one_out(svr_algr, X.values, y.values)

    # fit regression model to the data
    # model = svr_algr.fit(X, y.values.ravel())
    # 利用我們的model預測中選率
    predicted_y = model.predict(X) # predicted_y是一個雙層numpy array
    
    # 處理y和predicted_y的資料結構，以方便後續處理
    y = np.array(y)
    predicted_y = predicted_y.reshape(X.shape[0], 1)

    # 列印各個結果
    # print_y_and_predicted_y(y, predicted_y)
    print_r2_score(y, predicted_y)
    # print_coefficients(model, features)
    print_MSE(y, predicted_y)
    plot_true_and_pred_scatter(y, predicted_y)


def print_y_and_predicted_y(y, predicted_y):
    row_list = list()
    for index in range(len(y)):
        row_list.append([y[index][0], predicted_y[index][0]])
    df = pd.DataFrame(row_list, columns=['original_y', 'predicted_y'])
    # print df
    # print df.loc[df[['']], :] 

    print '-----------------------'


def print_coefficients(model, features):
    coefs = model.coef_[0]
    row_list = []
    row_list.append(('Intercept', model.intercept_[0]))
    for i in range(len(features)):
        row_list.append((features[i],coefs[i]))
    result =  pd.DataFrame(row_list, columns=['feature_name','coefficient'])
    print result.sort_values('coefficient', ascending=False)
    print '-----------------------'


def print_r2_score(y, predicted_y):
    print 'R-Square: ', r2_score(y, predicted_y)
    print '-----------------------'


def print_MSE(y, predicted_y):
    sum_y = reduce(lambda x,y: x+y, y)
    print 'y_true的平均值 ', (sum_y / len(y))[0]
    mse = mean_squared_error(y, predicted_y)
    print 'MSE ', mse
    print 'RMSE', math.sqrt(mse)
    print '-----------------------'


def plot_true_and_pred_scatter(y, predicted_y):
    fig, ax = plt.subplots()
    ax.scatter(y, predicted_y, s=10)
    ax.set_xlabel('True label', fontsize=20)
    ax.set_ylabel('Predicted label', fontsize=20)
    minEdge = min(y.min(),predicted_y.min())
    maxEdge = max(y.max(),predicted_y.max())
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.axis([minEdge, maxEdge, minEdge, maxEdge])
    plt.gcf().set_size_inches( (6, 6) )
    plt.show()

def standardizing(data):
    # data = data.reset_index(drop=True)
    X = data.iloc[:, :-1]
    X_array = StandardScaler().fit_transform(X)
    X_df = pd.DataFrame(X_array, columns=X.columns)
    result = pd.concat([X_df, data[['label']]], axis=1)
    return result

def normalizing(data):
    # data = data.reset_index(drop=True)
    X = data.iloc[:, :-1]
    X_array = MinMaxScaler().fit_transform(X)
    X_df = pd.DataFrame(X_array, columns=X.columns)
    result = pd.concat([X_df, data[['label']]], axis=1)
    return result

def standardizing_with_label(data):
    data = data.reset_index(drop=True)
    # X = data.iloc[:, :-1]
    X_array = StandardScaler().fit_transform(data)
    X_df = pd.DataFrame(X_array, columns=data.columns)
    # result = pd.concat([X_df, data[['label']]], axis=1)
    return X_df

def leave_one_out(algr, X, y):
    loo = LeaveOneOut()
    square_error_sum = 0.0
    for train_index, test_index in loo.split(X):
        # print "TRAIN:", train_index
        # print "TEST:", test_index
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print X_train
        model = algr.fit(X_train, y_train.ravel())
        predicted_y = model.predict(X_test)
        # print predicted_y
        # print y_test
        square_error_sum += float(y_test[0] - predicted_y) ** 2
        # print square_error_sum
        # break
    mse = square_error_sum / X.shape[0]
    print '-----------------------'
    print 'Leave One Out的mse ' , mse
    print '-----------------------'

def filter_look_time(data):
    data = data.drop(data[data['look_times'] < 21].index)
    data = data.reset_index(drop=True)
    return data

def drop_columns(data, index):
    print len(data.columns)
    print '===================> 移除', data.columns[index], '<====================='
    data = data.drop(data.columns[index], axis=1)
    # ridge_regression(data)
    SVR(data)

def drop_GID_looktime(data):
    data = data.drop(['GID', 'look_times', 'productFormatCount'], axis=1)
    return data

def activate_drop_columns(data):
    for index in range(1, len(data.columns)-2):
        drop_columns(data, index)

def scatter_plots(data):
    data.plot(x='productFormatCount', y='img_height', style='o')
    plt.show()

def join_new_label(data, label):
    data = data.drop(['label'], axis=1)
    # data = data.rename(columns={'label': 'label_old'})
    # print data
    result = pd.merge(data, label, left_on='GID', right_on='gid')
    result = result.drop(['gid'], axis=1)
    # print result

    # print result.sort_values('label_old', ascending=True)[['label_old']]
    # print result.sort_values('label', ascending=True)[['label']]

    return result


if __name__ == '__main__':    
    data = pd.read_csv('./result_detergent_898_4.csv')

    label = pd.read_csv('./ProValue_4.0.csv')
    # print label[['label']]

    data = filter_look_time(data)
    data = join_new_label(data, label)
    # data.to_csv('~/Downloads/new.csv')

    # print data[['label']].describe()
    # print data
    # print data.loc[data['GID'] == '2069836', :]
    data = drop_GID_looktime(data)
    # print data.columns

    # print data

    # data = standardizing(data)
    data = normalizing(data)
    # data = standardizing_with_label(data)
    print data.shape

    # scatter_plots(data)
    # activate_drop_columns(data)

    # linear_regression(data)
    ridge_regression(data)
    # SVR(data)
