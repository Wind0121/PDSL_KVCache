import matplotlib.pyplot as plt

# 创建数据
x = [1, 2, 3, 4, 5]

# 第一条折线数据
y1 = [1, 4, 9, 16, 25]

# 第二条折线数据
y2 = [1, 2, 3, 4, 5]

# 第三条折线数据
y3 = [2, 3, 5, 7, 11]

# 绘制多条折线
plt.plot(x, y1, label="y = x^2", marker='o')
plt.plot(x, y2, label="y = x", marker='s')
plt.plot(x, y3, label="y = primes", marker='^')

# 添加标题和轴标签
plt.title('Multiple Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图例
plt.legend()

# 保存图片
plt.savefig('multiple_lines_plot.png', dpi=300, bbox_inches='tight')