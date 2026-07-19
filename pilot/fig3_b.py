import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
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

N = 5
bar_width = 0.5
# bar_width_ndcg = 0.3
indexes = np.arange(N)
# hatches = ('', '**', '++', 'oo', '', '*', 'o', '.', 'O')
hatches_recall = '..'
# hatches_ndcg = '\\\\'
#hatches = ('', '+++', '///', '\\\\\\', '')
# colors = ['Red', 'Skyblue', 'Orange', 'LightGrey', 'MediumSlateBlue', 'Tomato', 'Palegreen', 'Azure']
# colors = ['Skyblue', 'Orange', 'Palegreen', 'Tomato', 'LightGrey']
colors = ['#96CAC1', '#F6F6BE', '#C1BFD7', '#EA8E84', 'LightGrey']
# colors = np.array([(244/256,0,0),(15/256,107/256,175/256),(19/256,167/256,43/256),(146/256,48/256,151/256),
                # (255/256,94/256,0/256),(255/256,255/256,0),(161/256,61/256,19/256),(255/256,92/256,181/256)])
# colors_ndcg = ['Skyblue', 'Orange', 'LightGrey', 'Tomato']

models = ['w/o Gen', 'Attentive', 'MLP', 'GNN', 'w/o Reg']

auc = [91.79, 92.87, 94.40, 95.48, 93.08]
std = [3.75, 1.50, 2.58, 1.70, 2.64]

figure = plt.figure()
fig,axis=plt.subplots()
axis.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
# right_axis = left_axis.twinx()
for i in range(N):
    rec_recall = axis.bar(indexes[i]+0.5, auc[i], width=bar_width, color=colors[i], edgecolor="Grey", linewidth=2, hatch=hatches_recall,
                yerr=std[i], error_kw={'ecolor':'crimson', 'elinewidth':3, 'capsize':5}, label=models[i])

axis.set_ylabel("AUC")
# axis.set_xlabel('Number of Factor Graphs')
#plt.yscale("log")
axis.grid(True, linestyle='-.', axis='y')
axis.set_xlim(0, N)
axis.set_ylim(83, 98)
# right_axis.set_ylim(0.25, 0.40)
axis.set_xticks(indexes+0.5)
axis.set_xticklabels(['w/o Gen', 'Attentive', 'MLP', 'GNN', 'w/o Reg'], rotation=20)
# axis.set_yticks([x/100.0 for x in range(5,10,1)], [x/100.0 for x in range(5,10,1)])
axis.set_yticks([83, 86, 89, 92, 95, 98])
# plt.xticks(fontname = "Times New Roman")
# plt.yticks(fontname = "Times New Roman")
plt.subplots_adjust(top=0.975, right=0.99, left=0.15, bottom=0.15)
# plt.subplots_adjust(top=0.98, right=0.975, left=0.13, bottom=0.09)
plt.savefig("./fig3_b.pdf")

plt.show()