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



def set_ax(ax, result, x_lable, y_label, z_label):
    colors = ['r', 'b', 'g', 'y', 'b', 'p']
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(x_lable, fontsize = 37, rotation = 0, labelpad=25)
    ax.set_ylabel(y_label, fontsize = 37, rotation = 0, labelpad=25)
    ax.set_zlabel(z_label, fontsize = 37, rotation = 90, labelpad=21)
    xlabels = np.array(['1', '2', '3', '4', '5'])
    xpos = np.array([0, 1, 2, 3, 4])
    ylabels = np.array(['0.1', '0.3', '0.5', '0.7', '0.9'])
    ypos = np.array([0, 1, 2, 3, 4])

    xposM, yposM = np.meshgrid(xpos, ypos, copy = False)
    
    # ax.yaxis.set_label_coords(-1, 0.5)
    zpos = result
    zpos = zpos.ravel()

    dx = 0.5
    dy = 0.5
    dz = zpos
    # zmin = np.min(result) - 1
    # zmax = np.max(result)+ 0.3
    # zlabels = np.arange(zmin, zmax, 0.7)  # 设置 z 轴刻度从 zmin 到 zmax，步长为 1

    zmin = np.min(result) - 0.5
    zmax = np.max(result)+ 0.5
    zlabels = np.arange(zmin, zmax, 0.5)  # 设置 z 轴刻度从 zmin 到 zmax，步长为 1
    bottom = np.ones_like(dz) * zmin

    dz = dz - bottom

    ax.w_xaxis.set_ticks(xpos + dx / 2.)
    ax.w_xaxis.set_ticklabels(xlabels)
    ax.w_yaxis.set_ticks(ypos + dy / 2.)
    ax.w_yaxis.set_ticklabels(ylabels)

    ax.set_zlim3d(zmin = zmin, zmax = zmax)
    ax.w_zaxis.set_ticks(zlabels)

    values = np.linspace(0.3, 0.9, xposM.ravel().shape[0])
    colors = cm.rainbow(values)
    ax.bar3d(xposM.ravel(), yposM.ravel(), bottom, dx, dy, dz, color = colors, alpha=0.65)
    ax.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

    ax.tick_params(axis = 'x', labelsize = 32, pad=5)
    ax.tick_params(axis = 'y', labelsize = 32, pad=5)
    ax.tick_params(axis = 'z', labelsize = 32, pad=8)
    
    ax.view_init(25, 30)
    
    
def plot_3d_bar(ndcg, fig_name):
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.default'] = 'regular'
    plt.rc('font', family = 'Times New Roman')
    fig = plt.figure(figsize = (10, 12)) 
    ax = fig.add_subplot(111, projection = '3d')
    
    # set_ax(ax, ndcg, r"$\mathcal{d}$", r"$\alpha$", r'$NDCG@10$')
    set_ax(ax, ndcg, r"$\it{d}$", r"$\alpha$", r'$NDCG@10$')
    plt.savefig(fig_name, bbox_inches = 'tight')

if __name__ == '__main__':
    # 每一行是第一个维度变化的取值 # 每一列是第二个维度变化的取值
    # game
    game_ndcg = np.array([[4.58, 4.76, 4.60, 4.26, 4.60],
                       [4.35, 4.47, 5.04, 4.25, 4.66],
                       [4.69, 4.69, 5.13, 4.70, 4.64],
                       [4.63, 4.58, 5.07, 3.97, 4.43],
                       [4.51, 4.40, 4.89, 3.41, 4.09]])

    #video
    video_ndcg = np.array([[6.49, 6.47, 6.45, 6.25, 6.36],
                       [6.67, 7.05, 7.32, 6.38, 6.34],
                       [6.45, 6.26, 6.53, 6.60, 6.49],
                       [6.70, 6.85, 7.37, 6.42, 6.23],
                       [7.28, 7.34, 6.85, 7.36, 7.21]])
    plot_3d_bar(game_ndcg, './similarity_game.pdf')
    plot_3d_bar(video_ndcg, './similarity_video.pdf')
    