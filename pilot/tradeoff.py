import matplotlib.pyplot as plt

# # C1
# x_Dance = [0, 0.7, 1.65, 2.84, 3.55, 4.80, 5.05]
# y_Dance = [77.05, 78.58, 79.32, 79.76, 79.76, 79.76, 79.8]

# x_var1 = [0.82, 1.15, 2.33, 3.37, 4.41]
# y_var1 = [78.03, 78.50, 78.77, 78.77, 78.90]

# x_var2 = [0.55, 0.72, 2.54, 3.72, 4.23]
# y_var2 = [78.40, 78.51, 78.80, 79.24, 79.28]

# x_var3 = [1.03, 1.3, 2.89, 3.01, 3.91, 4.29, 4.47]
# y_var3 = [79.38, 79.52, 79.63, 79.63, 79.85, 79.93, 79.89]

# x_var4 = [0.38, 2.17, 4.22, 5.49, 5.8]
# y_var4 = [78.3, 78.54, 78.84, 79.32, 79.41]

# # 绘图
# plt.figure(figsize=(5, 4))
# plt.plot(x_Dance, y_Dance, label='Dance', color='blue')
# plt.plot(x_var1, y_var1, label='var1', color='green')
# plt.plot(x_var2, y_var2, label='var2', color='orange')
# plt.plot(x_var3, y_var3, label='var3', color='violet')
# plt.plot(x_var4, y_var4, label='var4', color='red')
# # 坐标轴和标题
# plt.xlabel('Equalized odds (%)')
# plt.ylabel('ACC (%)')
# plt.title('C1')

# # 图例
# plt.legend(loc='lower left', framealpha=0.8)

# # 网格线（可选）
# plt.grid(True, linestyle='--', alpha=0.3)

# plt.tight_layout()
# # plt.show()
# plt.savefig("C1.png", bbox_inches = 'tight')




# C2
x_Dance = [0, 0.40, 0.53, 0.62, 0.94, 1.19, 2.04]
y_Dance = [77.26, 78.4, 79.98, 80.04, 80.07, 80.21, 80.32]

x_var1 = [2.33, 2.66, 2.78, 2.86]
y_var1 = [79.58, 79.65, 79.71, 79.72]

x_var2 = [2.00, 2.03, 2.61, 3.08, 3.34]
y_var2 = [79.42, 79.58, 79.84, 79.86, 80.04]

x_var3 = [1.28, 1.45, 1.45, 1.52, 2.02, 4.10]
y_var3 = [79.89, 79.89, 80.07, 80.12, 80.13, 80.58]

x_var4 = [1.65, 1.81, 1.99, 3.40]
y_var4 = [79.42, 79.74, 79.75, 80.01]

# 绘图
plt.figure(figsize=(5, 4))
plt.plot(x_Dance, y_Dance, label='Dance', color='blue')
plt.plot(x_var1, y_var1, label='var1', color='green')
plt.plot(x_var2, y_var2, label='var2', color='orange')
plt.plot(x_var3, y_var3, label='var3', color='violet')
plt.plot(x_var4, y_var4, label='var4', color='red')
# 坐标轴和标题
plt.xlabel('Equalized odds (%)')
plt.ylabel('ACC (%)')
plt.title('C2')
# 图例
plt.legend(loc='lower left', framealpha=0.8)

# 网格线（可选）
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
# plt.show()
plt.savefig("C2.png", bbox_inches = 'tight')



# # C3
# x_Dance = [0, 0.31, 1.71, 2.61, 3.32]
# y_Dance = [70.89, 72.77, 73.57, 74.27, 74.75]

# x_var1 = [1.86, 1.91, 2.04, 2.25, 2.72]
# y_var1 = [72.88, 72.90, 73.01, 73.22, 73.56]

# x_var2 = [1.37, 2.43, 2.78, 3.36]
# y_var2 = [71.96, 72.46, 72.48, 72.66]

# x_var3 = [2.68, 2.80, 3.94, 3.97, 4.11, 4.24]
# y_var3 = [72.27, 72.67, 72.76, 73.07, 73.18, 73.65]

# x_var4 = [2.03, 2.60, 2.88, 2.90, 3.81]
# y_var4 = [73.06, 74.22, 74.27, 74.59,74.64]

# # 绘图
# plt.figure(figsize=(5, 4))
# plt.plot(x_Dance, y_Dance, label='Dance', color='blue')
# plt.plot(x_var1, y_var1, label='var1', color='green')
# plt.plot(x_var2, y_var2, label='var2', color='orange')
# plt.plot(x_var3, y_var3, label='var3', color='violet')
# plt.plot(x_var4, y_var4, label='var4', color='red')
# # 坐标轴和标题
# plt.xlabel('Equalized odds (%)')
# plt.ylabel('ACC (%)')
# plt.title('C3')
# # 图例
# plt.legend(loc='lower left', framealpha=0.8)
# # 网格线（可选）
# plt.grid(True, linestyle='--', alpha=0.3)

# plt.tight_layout()
# # plt.show()
# plt.savefig("C3.png", bbox_inches = 'tight')


# # C4
# x_Dance = [0, 0.36, 0.42, 0.48, 0.58, 0.99, 1.54, 1.64]
# y_Dance = [70.86, 72.19, 73.07, 73.75, 73.98, 74.01, 74.25, 74.53]

# x_var1 = [1.89, 2.09, 2.15, 2.43, 2.76]
# y_var1 = [72.83, 72.92, 73.17, 73.38, 73.55]

# x_var2 = [0.7, 0.92, 1.29, 1.66, 1.86, 2.04]
# y_var2 = [70.96, 71.84, 72.35, 72.65, 72.68, 73.27]

# x_var3 = [1.38, 1.40, 1.60, 1.95, 2.01]
# y_var3 = [73.02, 73.95, 73.95, 74.18, 74.34]

# x_var4 = [1.24, 1.31, 1.50, 1.64]
# y_var4 = [72.76, 73.04, 73.24, 73.85]

# # 绘图
# plt.figure(figsize=(5, 4))
# plt.plot(x_Dance, y_Dance, label='Dance', color='blue')
# plt.plot(x_var1, y_var1, label='var1', color='green')
# plt.plot(x_var2, y_var2, label='var2', color='orange')
# plt.plot(x_var3, y_var3, label='var3', color='violet')
# plt.plot(x_var4, y_var4, label='var4', color='red')

# # 坐标轴和标题
# plt.xlabel('Equalized odds (%)')
# plt.ylabel('ACC (%)')
# plt.title('C4')

# # 图例
# plt.legend(loc='lower left', framealpha=0.8)

# # 网格线（可选）
# plt.grid(True, linestyle='--', alpha=0.3)

# plt.tight_layout()
# # plt.show()
# plt.savefig("C4.png", bbox_inches = 'tight')