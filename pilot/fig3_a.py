import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# import matplotlib.ticker as ticker

# plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
import matplotlib.ticker as mtick
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
# matplotlib.rcParams["figure.figsize"] = (8, 5)


fig, axis = plt.subplots()
axis.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
# axis.grid(color='darkgrey',linestyle='-', axis='y')
plt.grid(color='darkgrey',linestyle='-', axis='y', alpha=0.3) 
# plt.grid(zorder=0) # 让网格线置于底层，同时需要让线置于上层（zorder=100）

# AMAP
# & M1 & 67.07±3.93 & 62.57±1.89 & 48.56±3.35 & 59.39±5.35
# & M2 & 69.89±4.73 & 64.02±2.23 & 52.21±3.70 & 60.39±6.36
# & M3 & 73.01±4.07 & 65.64±2.13 & 54.47±3.64 & 66.60±5.72
# & M4 & 76.92±5.62 & 66.50±2.58 & 57.01±4.44 & 71.18±6.76 
# & M5 & 77.58±1.37 & 64.66±1.76 & 58.17±2.38 & 70.34±1.60
# & CoCo & 79.27±0.70 & 68.85±1.55 & 60.94±1.51 & 72.36±1.15

x = np.arange(4)
#dp
height1 = np.array([1.62, 3.50, 2.60, 2.35])
height2 = np.array([0.86, 2.69, 2.01, 1.56])
height3 = np.array([1.03, 1.62, 1.53, 1.27])
height4 = np.array([1.54, 1.87, 1.38, 2.30])
height5 = np.array([2.74, 2.54, 2.01, 1.52]) 


# height1 = np.array([1.50, 2.47, 1.70, 1.57])
# height2 = np.array([0.74, 2.62, 1.98, 1.34])
# height3 = np.array([0.92, 1.50, 1.49, 0.86])
# height4 = np.array([1.45, 1.99, 1.31, 1.33])
# height5 = np.array([1.88, 2.05, 1.83, 2.04]) 

plt.xlim((-0.15, 3.85))
plt.ylim((0, 4))
# 设置宽度值（偏移量）
width = 0.12
# 绘制不同数据时，x 轴依次增加一个偏移量
epsilon = 0.02
plt.bar(x,                           height1, width, color= '#8491B4CC',    label="$\delta$=0.1", edgecolor="none")
plt.bar(x + width + epsilon,         height2, width, color='#91D1C2CC', label="$\delta$=0.3", edgecolor="none")
plt.bar(x + width * 2 + epsilon * 2, height3, width, color='#3C5488CC',     label="$\delta$=0.5", edgecolor="none")
plt.bar(x + width * 3 + epsilon * 3, height4, width, color='#00A087CC',    label="$\delta$=0.7", edgecolor="none")
# plt.bar(x + width * 3 + epsilon * 3, height4, width, color='orange',        label="$\delta$=0.9", edgecolor="none")
plt.bar(x + width * 4 + epsilon * 4, height5, width, color=  '#4DBBD5CC',  label="$\delta$=0.9", edgecolor="none") 
# plt.bar(x + width * 5 + epsilon * 5, height6, width, color='darkseagreen',    label="K=15", edgecolor="none") 

# 设置 x 轴刻度的标签
plt.xticks(x + width*2, ['C1', 'C2','C3','C4'])

font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 26} # Times New Roman
plt.ylabel("Demographic Parity", font1)
# plt.ylabel("Equal Odds", font1)
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 26} # Times New Roman
plt.xlabel("The trade off parameter $\delta$ of Mixup", font2)

#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=25)
labels = axis.get_xticklabels() + axis.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]


font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18} # Times New Roman
# plt.legend(prop=font2, loc='upper left')
plt.legend(prop=font2, ncol=2, loc=1, frameon=True) # 横着放
# plt.legend(prop=font2, bbox_to_anchor=(0.5, 1.0), ncol=3, borderaxespad=0)

# index=np.arange(len(x))
# # for h in [height1, height2, height3, height4, height5]:
# for a, b in zip(index+0.1, height1):   #柱子上的数字显示
#     plt.text(a, b,'%.1f'%b,ha='center',va='bottom', fontsize=10)
# for a, b in zip(index+(width+epsilon)+0.1, height2):   #柱子上的数字显示
#     plt.text(a, b,'%.1f'%b,ha='center',va='bottom', fontsize=10)

ax = plt.gca() # 获取边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False) #去掉右边框
# axis.spines['top'].set_color("none")
ax.spines['left'].set_linewidth(0.5)
axis.spines['right'].set_linewidth(0.5)
axis.spines['bottom'].set_linewidth(0.5)
axis.spines['top'].set_linewidth(0.5)

plt.tight_layout()
plt.savefig("./delta_dp.pdf")
# plt.savefig("./delta_eo.pdf")
# plt.show()