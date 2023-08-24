import matplotlib.pyplot as plt

# 数据
x = [500, 1000, 2000, 3000, 4000, 5000, 10000]
y1 = [0.579, 0.586, 0.59, 0.590, 0.5845, 0.5851, 0.567]
y2 = [0.612, 0.621, 0.6231, 0.623, 0.617, 0.617, 0.6]

plt.figure(figsize=(8, 4))

# 绘制折线图
plt.plot(x, y1, marker='o', linestyle='-', label='Line 1')
plt.plot(x, y2, marker='o', linestyle='-', label='Line 2')

# 设置标题和坐标轴标签
plt.title('Performance Based On Data Size')
plt.xlabel('Data Size')
plt.ylabel('Precision')

# 显示网格线
plt.grid()
plt.xticks(x)
# 显示图例
plt.legend()
plt.savefig('NER_data_size.png', dpi=300)
# 显示图形
plt.show()
