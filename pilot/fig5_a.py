import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['axes.edgecolor'] = 'Grey'
# params={'font.family':'serif',
#         'font.serif':'Times New Roman',
#         'font.style':'normal',
#         'font.weight':'normal', #or 'blod'
#         }
# matplotlib.rcParams.update(params)
# matplotlib.rcParams['axes.titlesize'] = 10
# matplotlib.rcParams['figure.titleweight'] = 'bold'

# font = matplotlib.font_manager.FontProperties(fname='/home/wyf/SimHei.ttf')
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18

#matplotlib.rcParams['hatch.linewidth'] = 0.1

# matplotlib.rcParams['legend.fontsize'] = 7
# matplotlib.rcParams['legend.edgecolor'] = 'Grey'

N = 6
bar_width = 0.5
# bar_width_ndcg = 0.3
# indexes = np.arange(N)
# hatches = ('', '**', '++', 'oo', '', '*', 'o', '.', 'O')
# hatches_recall = '..'
# hatches_ndcg = '\\\\'
#hatches = ('', '+++', '///', '\\\\\\', '')
# colors = ['Red', 'Skyblue', 'Orange', 'LightGrey', 'MediumSlateBlue', 'Tomato', 'Palegreen', 'Azure']
# colors = ['Skyblue', 'Orange', 'Tomato', 'Palegreen', 'LightGrey']
# colors_ndcg = ['Skyblue', 'Orange', 'LightGrey', 'Tomato']
iters = list(range(0, 7))
# colors = ['mediumblue', 'maroon']
colors = ['#0079B0', '#B5D5E6']


# models = ['1', '2', '4', '8', '16']

acc = [92.34, 92.77, 95.31, 96.06, 97.91, 97.75, 97.55]
std = [2.20, 1.63, 1.39, 1.31, 0.70, 0.97, 0.90]


def draw_line(avg, std):
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))#下方差
    plt.plot(iters, avg, "-o", linewidth = 2, markersize=8, color=colors[0], label='Mean')
    plt.fill_between(iters, r1, r2, color=colors[1], alpha=0.6, label='STD')
    # plt.scatter(iters, avg, marker='>', edgecolors=colors[1], s=150, linewidth = 2)

fig, axis=plt.subplots()
axis.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

draw_line(acc, std)

axis.set_ylabel("AUC")
axis.set_xlabel('Number of Filter Graphs $N$')
#plt.yscale("log")
axis.grid(True, linestyle='-.', axis='both')
# axis.set_xlim(0, N)
axis.set_ylim(89, 99)
# right_axis.set_ylim(0.25, 0.40)
axis.set_yticks([89, 91, 93, 95, 97, 99])
plt.xticks(iters,[1, 2, 4, 8, 16, 32, 64])
axis.legend(loc=4, ncol=1, prop=font1)
plt.subplots_adjust(top=0.975, right=0.99, left=0.15, bottom=0.135)
plt.savefig("./fig4_a.pdf")

plt.show()