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

def Y_preY_histogram(self):
    
