import matplotlib
import numpy as np
from sklearn.manifold import TSNE
# import torch
# import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn
# import torch.nn.functional as F
import os
from matplotlib import pyplot
from scipy.interpolate import make_interp_spline
import seaborn as sns
from matplotlib import cm

from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker
import matplotlib.ticker as mtick

def plot_3d_bar(result, name):
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.default'] = 'regular'
    
    result = np.array(result).transpose()
    
    colors = ['r', 'b', 'g', 'y', 'b', 'p']
    fig = plt.figure() # figsize = (4.5, 2.6)
    ax1 = fig.add_subplot(111, projection = '3d')

    ax1.set_xlabel("Metric", fontsize = 12)
    ax1.set_ylabel('I', fontsize = 12)
    ax1.set_zlabel('Score', fontsize = 12)
    xlabels = np.array(['ACC', 'F1', 'NMI', 'ARI'])
    # xpos = np.arange(xlabels.shape[0])
    xpos = np.array([0,1,2,3])
    ylabels = np.array(['0', '1', '2', '3', '4', '5', '6'])
    # ypos = np.arange(ylabels.shape[0])
    ypos = np.array([0,1,2,3,4,5,6])

    xposM, yposM = np.meshgrid(xpos, ypos, copy = False)

    zpos = result
    zpos = zpos.ravel()

    dx = 0.5
    dy = 0.5
    dz = zpos

    zmin = np.min(result)-7
    zmax = np.max(result)+7

    bottom = np.ones_like(dz) * zmin

    dz = dz - bottom

    ax1.w_xaxis.set_ticks(xpos + dx / 2.)
    ax1.w_xaxis.set_ticklabels(xlabels)

    ax1.w_yaxis.set_ticks(ypos + dy / 2.)
    ax1.w_yaxis.set_ticklabels(ylabels)

    ax1.set_zlim3d(zmin = zmin, zmax = zmax)

    values = np.linspace(0.3, 0.9, xposM.ravel().shape[0])
    colors = cm.rainbow(values)
    ax1.bar3d(xposM.ravel(), yposM.ravel(), bottom, dx, dy, dz, color = colors)
    ax1.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

    ax1.tick_params(axis = 'x', labelsize = 10)
    ax1.tick_params(axis = 'y', labelsize = 10)
    ax1.tick_params(axis = 'z', labelsize = 10)

    ax1.view_init(35, -30) # 旋转

    fig.set_size_inches(5, 3)

    fig_name = name + '.pdf'
    save_path = os.path.join('./', fig_name)
    plt.savefig(save_path, bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    # 每一行是第一个维度变化的取值 # 每一列是第二个维度变化的取值
    # cora
    result = np.array([[54.76, 78.66, 79.06, 79.09, 79.17, 79.23, 79.25],
                       [52.74, 73.82, 76.59, 76.95, 76.46, 76.78, 76.41],
                       [33.65, 59.22, 60.68, 60.90, 61.05, 61.26, 60.93],
                       [26.13, 58.36, 59.60, 59.84, 59.48, 60.26, 59.71]])
    name = 'fig4_a'
    plot_3d_bar(result, name)

    # citeseer
    # result = np.array([[57.56, 71.69, 71.96, 72.16, 72.05, 72.28, 72.11],
    #                    [54.42, 63.28, 65.77, 65.82, 66.26, 66.42, 66.33],
    #                    [29.95, 44.78, 44.99, 45.12, 45.29, 45.38, 45.17],
    #                    [28.86, 47.10, 48.19, 48.32, 48.43, 48.65, 48.30]])
    # name = 'fig4_b'
    # plot_3d_bar(result, name)

    # AMAP
    # result = np.array([[76.71, 77.14, 77.76, 78.21, 78.90, 79.06, 78.67],
    #                    [71.14, 71.89, 71.99, 72.10, 72.80, 72.97, 72.51],
    #                    [65.28, 66.02, 67.00, 67.26, 67.67, 67.79, 67.59],
    #                    [56.87, 57.05, 58.92, 59.62, 60.35, 60.98, 60.78]])
    # name = 'fig4_c'
    # plot_3d_bar(result, name)

    # BAT
    # result = np.array([[78.85, 80.84, 80.92, 80.84, 80.92, 80.92, 80.84],
    #                    [78.77, 80.77, 80.81, 80.67, 80.75, 80.89, 80.69],
    #                    [54.96, 57.41, 57.86, 57.76, 58.08, 58.11, 58.05],
    #                    [53.12, 56.57, 57.06, 56.95, 57.31, 57.17, 57.00]])
    # name = 'fig4_d'
    # plot_3d_bar(result, name)

    # EAT
    # result = np.array([[57.89, 58.02, 58.15, 58.22, 58.47, 58.55, 58.30],
    #                    [57.77, 57.97, 58.03, 58.16, 58.46, 58.54, 58.24],
    #                    [33.59, 33.81, 34.11, 34.18, 34.34, 34.45, 34.12],
    #                    [27.33, 27.61, 27.64, 28.09, 28.16, 28.30, 28.05]])
    # name = 'fig4_e'
    # plot_3d_bar(result, name)

    # EAT
    # result = np.array([[58.49, 58.99, 59.08, 60.76, 60.86, 61.11, 60.47],
    #                    [54.08, 57.82, 58.58, 59.29, 60.06, 59.71, 59.31],
    #                    [27.20, 27.77, 28.19, 28.49, 28.34, 29.53, 29.24],
    #                    [28.07, 28.30, 28.58, 29.16, 28.76, 29.43, 29.11]])
    # name = 'fig4_f'
    # plot_3d_bar(result, name)

    