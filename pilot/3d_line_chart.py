import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
import os
from mpl_toolkits.mplot3d import axes3d

def plot_ax(ax, result):
   x = np.arange(0.1, 0.6, 0.1)
   y = np.arange(0.1, 0.6, 0.1)

   X, Y = np.meshgrid(x, y)

   Z = result
   Z = np.array(Z).T

   ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('Blues'), linewidth=0.5, antialiased=True, edgecolor= 'gray', alpha=0.8)
   ax.set_xticks(np.arange(0.1, 0.6, 0.1))
   ax.set_yticks(np.arange(0.1, 0.6, 0.1))
   # ax.set_zticks(np.arange(63, 68, 1))
   ax.set_zticks(np.arange(71, 73.5, 0.5))

   xlabels = np.array(['0.2', '0.4', '0.6','0.8', '1'])
   ylabels = np.array(['0.2', '0.4', '0.6', '0.8', '1'])
   # zlabels = np.array(['63', '64', '65', '66', '67'])
   zlabels = np.array(['71', '71.5', '72.0', '72.5', '73.0'])
   ax.w_xaxis.set_ticks(x)
   ax.w_xaxis.set_ticklabels(xlabels)
   ax.w_yaxis.set_ticks(y)
   ax.w_yaxis.set_ticklabels(ylabels)
   ax.w_zaxis.set_ticklabels(zlabels)
   
   ax.xaxis.set_rotate_label(False)
   ax.yaxis.set_rotate_label(False)
   ax.zaxis.set_rotate_label(False)
   ax.set_xlabel(r'$\beta$', fontdict = dict(fontsize=37), rotation = 0, labelpad=25)
   ax.set_ylabel(r'$\lambda$', fontdict = dict(fontsize=37), rotation = 0, labelpad=25)
   ax.set_zlabel('Accuracy', fontdict = dict(fontsize=37), rotation = 90, labelpad=21)
   ax.tick_params(axis = 'x', labelsize = 32, pad=5)
   ax.tick_params(axis = 'y', labelsize = 32, pad=5)
   ax.tick_params(axis = 'z', labelsize = 32, pad=8)
   ax.view_init(25, 30)

plt.rc('font', family = 'Times New Roman')

fig = plt.figure(figsize = (10, 12))

# ax1 = fig.add_subplot(111, projection = '3d')
# B1_ndcg = np.array([[65.86, 65.70, 65.77, 65.74, 65.42], #lambda 0.2
#                        [65.87, 65.69, 65.83, 65.39, 65.71],#lambda 0.4
#                        [65.90, 66.58, 66.48, 64.88, 64.21],#lambda 0.6
#                        [64.12, 64.99, 65.32, 65.77, 64.55],#lambda 0.8
#                        [63.95, 64.37, 64.67, 63.11, 63.98]])#lambda 1
# plot_ax(ax1, B1_ndcg.T)
# fig_name = './factors_B1.pdf'

ax2 = fig.add_subplot(111, projection = '3d')
B2_ndcg = np.array([[72.60, 72.40, 72.39, 72.49, 72.43],    #lambda 0.2
                       [72.09, 72.82, 72.57, 72.26, 72.20], #lambda 0.4
                       [71.78, 72.88 , 72.80, 72.71, 72.68],#lambda 0.6
                       [71.63, 72.65, 72.19, 71.80, 72.35], #lambda 0.8
                       [72.00, 72.03, 71.83, 71.78, 71.89]])#lambda 1
plot_ax(ax2, B2_ndcg.T)
fig_name = './factor_B2.pdf'

plt.savefig(fig_name, bbox_inches = 'tight')
# plt.show()