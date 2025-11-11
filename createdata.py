import csv
import random

# 生成随机对象和得分数据
data = [(f"对象{i}", random.uniform(0, 60)) for i in range(1, 201)]
#tu 40 text 25
# 将数据写入CSV文件
with open(r'D:\plus\sha\result.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['对象', '夏普利值']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for item in data:
        writer.writerow({'对象': item[0], '夏普利值': item[1]})


#
# #生成用户对物品评分
# import numpy as np
#
# # 生成随机矩阵
# user_item_matrix = np.random.randint(0, 11, size=(7, 200))
#
# # 转置矩阵
# user_item_matrix_transposed = user_item_matrix.T
#
# # 将每一行数据包含在一个列表中，并在两端加上方括号，每行最后加上逗号
# rows_with_brackets = [f"[{', '.join(map(str, row))}]," for row in user_item_matrix_transposed]
#
# # 保存为CSV文件
# np.savetxt('transposed_matrix.csv', rows_with_brackets, delimiter='', fmt='%s', newline='\n', header='', footer='', comments='')
#

# import networkx as nx
# import matplotlib.pyplot as plt
#
# # 创建一个图
# G = nx.Graph()
#
# # 添加节点
# G.add_node(1)
# G.add_node(2)
# G.add_node(3)
#
# # 添加边
# G.add_edge(1, 2)
# G.add_edge(2, 3)
#
# # 使用力导向布局算法布局节点
# pos = nx.spring_layout(G)
#
# # 绘制图
# nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=12, font_color='black', edge_color='gray')
#
# # 显示图
# plt.show()





# import pandas as pd
#
# # 创建示例数据
# data = {
#     '对象': ['对象1', '对象2', '对象3'],
#     '夏普利值': ['4.0', '3.5', '2.5']
# }
#
# # 创建DataFrame
# shape_values = pd.DataFrame(data)
#
# # 提取数字并转换为浮点数
# shape_values['夏普利值'] = shape_values['夏普利值'].str.extract('(\d+\.\d+)').astype(float)
#
# # 输出转换后的DataFrame
# print(shape_values)
