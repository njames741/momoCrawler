
# -*- coding: utf-8 -*-
import math
from pprint import pprint
import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import svm
import pickle
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from plotting import plot_true_and_pred_scatter

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_rows', 1000)


def ridge_regression(data, a):
    features = data.columns.tolist()
    features.remove('label')
    response = ['label']
    # 宣告一個Ridge Regression model
    lr = Ridge(alpha=a)
    # 定義應變數: label(需是一個DataFrame)
    y = data[response]
    # 定義features (需是一個DataFrame)
    X = data[features]

    _leave_one_out(lr, X.values, y.values)
    
    # fit regression model to the data
    model = lr.fit(X, y)
    # 利用我們的model預測中選率
    predicted_y = model.predict(X) # predicted_y是一個雙層numpy array
    # 把原本y的DataFrame也轉成雙層numpy array，方便等等列印
    y = np.array(y)

    # 列印各個結果
    _print_y_and_predicted_y_and_corr(y, predicted_y)
    _print_r2_score(y, predicted_y)
    _print_coefficients(model, features, '~/Desktop/temp0830/body_WI_lt10.csv')
    _print_MSE(y, predicted_y)
    plot_true_and_pred_scatter(y, predicted_y)

def SVR(data):
    features = data.columns.tolist()
    features.remove('label')
    response = ['label']

    y = data[response]
    X = data[features]

    svr_algr = svm.SVR(C=1.0, kernel='rbf')
    # fit regression model to the data
    model = svr_algr.fit(X, y.values.ravel())

    _leave_one_out(svr_algr, X.values, y.values)

    # fit regression model to the data
    # 利用我們的model預測中選率
    predicted_y = model.predict(X) # predicted_y是一個雙層numpy array
    
    # 處理y和predicted_y的資料結構，以方便後續處理
    y = np.array(y)
    predicted_y = predicted_y.reshape(X.shape[0], 1)

    # 列印各個結果
    # _print_y_and_predicted_y_and_corr(y, predicted_y)
    _print_r2_score(y, predicted_y)
    _print_MSE(y, predicted_y)
    plot_true_and_pred_scatter(y, predicted_y)


def _print_y_and_predicted_y_and_corr(y, predicted_y):
    row_list = list()
    for index in range(len(y)):
        row_list.append([y[index][0], predicted_y[index][0]])
    df = pd.DataFrame(row_list, columns=['original_y', 'predicted_y'])
    print 'y 與 predicted_y 的 correlation 為 ', df.corr().values[0][1]
    print '-----------------------'

def _print_coefficients(model, features, output_path):
    coefs = model.coef_[0]
    row_list = []
    row_list.append(('Intercept', model.intercept_[0]))
    for i in range(len(features)):
        row_list.append((features[i],coefs[i]))
    result =  pd.DataFrame(row_list, columns=['feature_name','coefficient'])
    _weightProcessing(result) # 印正規化權重用的
    print '-----------------------'
    # print result.sort_values('coefficient', ascending=False)
    # result.to_csv(output_path, index=False)
    print '-----------------------'

def _print_r2_score(y, predicted_y):
    print 'R-Square: ', r2_score(y, predicted_y)
    print '-----------------------'


def _print_MSE(y, predicted_y):
    sum_y = reduce(lambda x,y: x+y, y)
    print 'y_true的平均值 ', (sum_y / len(y))[0]
    mse = mean_squared_error(y, predicted_y)
    print 'MSE ', mse
    print 'RMSE', math.sqrt(mse)
    print '-----------------------'

def standardizing(data):
    X = data.iloc[:, :-1]
    X_array = StandardScaler().fit_transform(X)
    X_df = pd.DataFrame(X_array, columns=X.columns)
    result = pd.concat([X_df, data[['label']]], axis=1)
    return result

def normalizing(data):
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

def _leave_one_out(algr, X, y):
    loo = LeaveOneOut()
    square_error_sum = 0.0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = algr.fit(X_train, y_train.ravel())
        predicted_y = model.predict(X_test)
        square_error_sum += float(y_test[0] - predicted_y) ** 2
    mse = square_error_sum / X.shape[0]
    print '-----------------------'
    print 'Leave One Out的mse ' , mse
    print '-----------------------'

def filter_look_time(data, threshold):
    data = data.drop(data[data['look_times'] < threshold].index)
    data = data.reset_index(drop=True)
    return data

def drop_columns(data, index):
    print len(data.columns)
    print '===================> 移除', data.columns[index], '<====================='
    data = data.drop(data.columns[index], axis=1)
    # ridge_regression(data)
    SVR(data)

def drop_something(data, drop_list):
    data = data.drop(drop_list, axis=1)
    return data

def activate_drop_columns(data):
    for index in range(1, len(data.columns)-2):
        drop_columns(data, index)

def scatter_plots(data):
    data.plot(x='productFormatCount', y='img_height', style='o')
    plt.show()

def join_new_label_and_price(data, new_price_and_label):
    data = data.drop(['label', 'price'], axis=1) # 丟掉舊label和price
    result = pd.merge(data, new_price_and_label, on='GID') # join新label
    column_list = result.columns.tolist() # 改變column順序，把price拿到第二個
    column_list = column_list[:1] + column_list[-1:] + column_list[1:-1]
    result = result[column_list]
    return result

def join_label(data, label):
    data = data.drop('label', axis=1) # 丟掉舊label和price
    result = pd.merge(data, label, left_on='GID', right_on='gid') # join新label
    result = result.drop(['gid'], axis=1) # 丟掉多出來的gid
    return result

def join_price_for_CR(data, new_price_and_label):
    data = data.drop(['price'], axis=1) # 丟掉舊price
    result = pd.merge(data, new_price_and_label[['GID', 'price']], on='GID') # join新label
    result = result[['GID', 'price', 'unitPrice', 'haveOrigin', 'volume', 'supplementary',
       'bottle', 'combination', 'payment_ConvenienceStore', 'img_height', 'is_warm', 'is_cold', '12H',
       'haveVideo', 'installments', 'look_times', 'label']]
    return result

def _weightProcessing(weightDF):
    weightDF = weightDF.loc[1:, :]
    weightDF['coefficient'] = weightDF['coefficient'].abs()
    min_max_scaler = preprocessing.MinMaxScaler()
    weight_scaled = min_max_scaler.fit_transform(weightDF[['coefficient']])
    weightDF['coefficient'] = weight_scaled
    print weightDF.sort_values('coefficient', ascending=False).to_string(index=False)
