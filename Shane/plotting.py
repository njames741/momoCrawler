# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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

def std_error(y, predicted_y):
    std_error_list = []
    for index in range(0, len(y)):
        std_error_list.append((np.absolute(y[index] - predicted_y[index]) / y[index])[0])
    std_error_list = np.asarray(std_error_list)
    
    # print std_error_list.max()
    count = 0
    threshold = 0.20
    for item in std_error_list:
        if item <= threshold:
            count += 1
    print '全部資料共', len(std_error_list), '筆'
    print '殘差在', threshold, '以下共', count, '筆'
    print '殘差在', threshold ,'以下的比例', count / float(len(std_error_list))
    print '殘差的平均', std_error_list.mean()
    print '殘差的標準差', std_error_list.std()

    plt.hist(std_error_list, rwidth=0.7)
    plt.show()


