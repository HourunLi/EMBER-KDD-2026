import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as mtick

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

# 示例数据
beta = np.array([0.2, 0.4, 0.6, 0.8, 1])

c1_acc = np.array([1.21, 1.07, 0.74, 0.66, 0.89])
c2_acc = np.array([2.04, 1.67, 1.62, 1.77, 1.91])
c3_acc = np.array([1.18, 1.03, 1.38, 1.27, 1.40])
c4_acc = np.array([1.69, 1.33, 1.24, 1.46, 1.32])

# c1_eo = np.array([0.42, 0.37, 0.70, 0.89, 0.88])
# c2_eo = np.array([0.83, 0.67, 0.40, 0.43, 0.72])
# c3_eo = np.array([0.58, 0.36, 0.31, 0.68, 0.55])
# c4_eo = np.array([0.58, 0.47, 0.36, 0.89, 0.78])

fig, axis = plt.subplots()
axis.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
plt.grid(color='darkgrey',linestyle='-', axis='y', alpha=0.6) 
# plt.rc('font', family = 'Times New Roman')
plt.figure(figsize=(7, 5.48))


font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30} # Times New Roman
# plt.ylabel("Demographic Parity", font1)
# plt.ylabel("Equal Odds", font1)
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20} # Times New Roman
# plt.xlabel("The trade off parameter $\delta$ of Mixup", font2)


# # 绘制折线
plt.plot(beta, c1_acc, marker='v', linestyle='-', color='lightgreen', label='C1', markersize=20, markeredgecolor='black', markeredgewidth=0.05) 
plt.plot(beta, c2_acc, marker='o', linestyle='-', color='peachpuff', label='C2', markersize=20, markeredgecolor='black', markeredgewidth=0.05)
plt.plot(beta, c3_acc, marker='s', linestyle='-', color='lightblue', label='C3', markersize=20, markeredgecolor='black', markeredgewidth=0.05)
plt.plot(beta, c4_acc, marker='*', linestyle='-', color='gold', label='C4', markersize=25, markeredgecolor='black', markeredgewidth=0.05)
# 添加标题和轴标签
# plt.xlabel(r'The ratio of synthesized nodes $\it{r}$', fontsize = 40, labelpad = 20)
plt.xlabel(r'The ratio of synthesized nodes $\gamma$', font1)
# plt.ylabel('Demographic Parity', fontsize = 40)
plt.ylabel('Demographic Parity', font1)
plt.legend(prop=font2, ncol=2, loc=3, framealpha=0.5)
plt.xticks([0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0.2, 0.6, 1.0, 1.4, 1.8, 2.2])
plt.tick_params(axis='both', which='major', labelsize=25) 
plt.tight_layout()
plt.savefig("./r_ratio_dp.pdf", bbox_inches = 'tight')


# plt.plot(beta, c1_eo, marker='v', linestyle='-', color='lightgreen', label='C1', markersize=20, markeredgecolor='black', markeredgewidth=0.03) 
# plt.plot(beta, c2_eo, marker='o', linestyle='-', color='peachpuff', label='C2', markersize=20, markeredgecolor='black', markeredgewidth=0.03)
# plt.plot(beta, c3_eo, marker='s', linestyle='-', color='lightblue', label='C3', markersize=20, markeredgecolor='black', markeredgewidth=0.03)
# plt.plot(beta, c4_eo, marker='*', linestyle='-', color='gold', label='C4', markersize=25, markeredgecolor='black', markeredgewidth=0.03)


# # plt.xlabel(r'The ratio of synthesized nodes $\it{r}$', fontsize = 40, labelpad = 20)
# plt.xlabel(r'The ratio of synthesized nodes $\gamma$', font1)
# plt.ylabel(r'Equal Odds', font1)
# # plt.ylabel(r'Equal Odds', fontsize = 40)
# # plt.legend(prop=font2, loc='upper right', framealpha=0.5)
# plt.legend(prop=font2, ncol=2, loc=3, framealpha=0.5)
# plt.xticks([0.2, 0.4, 0.6, 0.8, 1])
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# # plt.ylim(4, 7.5)
# plt.tick_params(axis='both', which='major', labelsize=25) 
# plt.tight_layout()
# plt.savefig("./r_ratio_eo.pdf", bbox_inches = 'tight')