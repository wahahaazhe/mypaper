import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# 用户评分矩阵
user_item_matrix = np.array([
[2, 0, 10, 1, 10, 6, 3],
[3, 4, 5, 9, 6, 4, 10],
[0, 2, 7, 6, 9, 4, 3],
[8, 1, 1, 2, 7, 9, 9],
[8, 2, 7, 10, 9, 0, 1],
[7, 1, 0, 4, 0, 2, 1],
# [5, 9, 3, 3, 10, 6, 10],
# [8, 5, 5, 6, 4, 4, 1],
# [7, 4, 2, 5, 1, 0, 3],
# [2, 7, 4, 3, 4, 3, 4],
# [0, 9, 4, 1, 6, 5, 4],
# [6, 0, 10, 9, 2, 5, 8],
# [5, 4, 3, 4, 1, 5, 10],
# [4, 7, 0, 4, 5, 2, 10],
# [1, 10, 3, 10, 9, 1, 8],
# [1, 8, 1, 9, 3, 7, 9],
# [2, 0, 8, 8, 8, 7, 5],
# [4, 2, 7, 1, 0, 6, 9],
# [9, 1, 6, 0, 8, 8, 3],
# [10, 1, 9, 10, 7, 0, 7],
# [8, 2, 5, 7, 0, 3, 7],
# [8, 2, 0, 5, 3, 6, 6],
# [0, 2, 7, 6, 8, 4, 3],
# [7, 5, 6, 2, 0, 3, 5],
# [4, 6, 8, 3, 5, 9, 0],
# [8, 1, 4, 5, 1, 2, 0],
# [3, 7, 4, 4, 4, 5, 1],
# [6, 0, 6, 4, 0, 10, 7],
# [8, 9, 10, 10, 0, 1, 0],
# [0, 7, 2, 7, 5, 5, 3],
# [6, 8, 3, 7, 10, 4, 7],
# [2, 9, 6, 1, 7, 1, 6],
# [9, 3, 1, 9, 2, 1, 0],
# [8, 6, 7, 5, 5, 2, 3],
# [1, 10, 1, 0, 1, 5, 6],
# [6, 5, 1, 6, 6, 7, 7],
# [9, 7, 6, 7, 1, 4, 9],
# [4, 5, 5, 0, 0, 10, 6],
# [7, 7, 6, 1, 7, 6, 10],
# [7, 1, 9, 6, 8, 7, 6],
# [7, 1, 4, 9, 1, 7, 0],
# [4, 7, 2, 2, 10, 10, 1],
# [2, 4, 4, 1, 5, 1, 4],
# [10, 3, 6, 2, 4, 5, 7],
# [1, 1, 2, 4, 7, 10, 9],
# [3, 6, 5, 10, 3, 1, 3],
# [6, 8, 5, 3, 4, 8, 6],
# [4, 4, 1, 0, 9, 2, 1],
# [8, 8, 0, 3, 2, 10, 8],
# [3, 6, 7, 0, 2, 10, 0],
# [9, 0, 3, 8, 6, 2, 4],
# [7, 9, 8, 10, 1, 5, 9],
# [1, 8, 2, 0, 0, 9, 9],
# [6, 4, 4, 9, 0, 2, 8],
# [0, 2, 8, 5, 4, 2, 5],
# [7, 3, 0, 6, 8, 10, 4],
# [5, 8, 4, 1, 3, 8, 7],
# [6, 8, 6, 4, 7, 9, 9],
# [8, 3, 10, 2, 6, 0, 5],
# [4, 0, 2, 5, 4, 0, 7],
# [9, 1, 8, 5, 3, 5, 5],
# [3, 6, 4, 8, 0, 2, 3],
# [5, 2, 8, 6, 10, 4, 5],
# [2, 0, 7, 4, 1, 4, 4],
# [7, 3, 10, 4, 7, 1, 4],
# [0, 1, 5, 3, 5, 3, 8],
# [5, 2, 5, 0, 3, 1, 8],
# [10, 4, 5, 2, 5, 1, 0],
# [0, 6, 2, 7, 2, 1, 7],
# [2, 0, 2, 4, 10, 6, 8],
# [7, 6, 6, 1, 1, 6, 9],
# [5, 0, 6, 8, 2, 7, 4],
# [2, 8, 10, 2, 6, 3, 3],
# [5, 3, 6, 2, 3, 7, 8],
# [5, 7, 7, 8, 3, 9, 3],
# [3, 0, 0, 8, 6, 5, 1],
# [9, 0, 10, 2, 7, 6, 7],
# [1, 8, 1, 10, 2, 7, 4],
# [10, 8, 5, 8, 9, 10, 4],
# [3, 6, 2, 9, 7, 3, 0],
# [9, 8, 2, 4, 5, 5, 10],
# [2, 9, 1, 10, 0, 0, 1],
# [4, 8, 9, 10, 0, 4, 1],
# [4, 9, 9, 2, 0, 10, 9],
# [9, 7, 8, 9, 10, 10, 2],
# [6, 10, 9, 10, 6, 8, 5],
# [6, 7, 6, 4, 1, 7, 9],
# [8, 2, 0, 5, 9, 3, 0],
# [4, 7, 0, 0, 9, 7, 10],
# [9, 2, 1, 7, 7, 2, 8],
# [9, 6, 8, 7, 8, 7, 2],
# [8, 3, 6, 9, 9, 7, 1],
# [6, 8, 1, 2, 8, 10, 8],
# [4, 6, 0, 5, 2, 2, 6],
# [4, 0, 2, 4, 6, 7, 1],
# [4, 6, 1, 8, 8, 3, 3],
# [5, 8, 5, 3, 4, 6, 1],
# [6, 6, 0, 6, 3, 3, 1],
# [0, 5, 8, 5, 3, 0, 3],
# [8, 1, 8, 4, 10, 2, 4],
# [10, 8, 2, 9, 8, 3, 10],
# [0, 3, 4, 6, 5, 8, 3],
# [9, 2, 1, 1, 3, 1, 4],
# [0, 8, 3, 6, 2, 2, 1],
# [10, 3, 7, 4, 1, 2, 6],
# [10, 0, 9, 3, 6, 5, 6],
# [4, 8, 1, 4, 4, 6, 1],
# [5, 0, 8, 6, 8, 1, 5],
# [0, 6, 1, 1, 4, 0, 0],
# [0, 7, 8, 3, 3, 8, 5],
# [5, 7, 7, 9, 9, 5, 1],
# [0, 9, 3, 0, 0, 7, 5],
# [2, 3, 6, 6, 7, 0, 2],
# [2, 2, 3, 10, 1, 1, 2],
# [1, 4, 10, 6, 3, 5, 3],
# [5, 1, 7, 4, 7, 1, 8],
# [8, 7, 7, 8, 10, 7, 10],
# [4, 3, 5, 10, 4, 0, 6],
# [2, 4, 6, 8, 0, 5, 6],
# [5, 0, 10, 9, 4, 8, 7],
# [6, 9, 9, 3, 6, 10, 10],
# [5, 1, 7, 7, 1, 7, 3],
# [3, 4, 3, 6, 4, 4, 5],
# [3, 6, 5, 10, 1, 1, 8],
# [2, 4, 1, 5, 10, 2, 2],
# [6, 10, 7, 1, 10, 3, 8],
# [3, 7, 9, 1, 9, 0, 5],
# [7, 5, 1, 4, 3, 2, 8],
# [0, 4, 1, 1, 10, 10, 10],
# [5, 4, 1, 10, 1, 4, 5],
# [5, 5, 1, 1, 6, 6, 10],
# [4, 7, 10, 2, 3, 6, 10],
# [4, 0, 6, 2, 7, 0, 10],
# [5, 2, 9, 6, 0, 5, 0],
# [5, 5, 2, 8, 3, 8, 8],
# [9, 2, 4, 7, 5, 2, 8],
# [7, 7, 9, 0, 9, 0, 10],
# [4, 1, 7, 2, 5, 0, 0],
# [3, 2, 3, 9, 1, 9, 1],
# [5, 3, 3, 8, 1, 0, 10],
# [8, 3, 3, 6, 6, 9, 1],
# [7, 6, 0, 4, 7, 8, 3],
# [2, 7, 1, 6, 4, 9, 6],
# [2, 0, 1, 4, 3, 2, 8],
# [0, 3, 10, 4, 4, 8, 0],
# [0, 7, 10, 7, 9, 3, 10],
# [1, 7, 8, 6, 4, 6, 10],
# [2, 1, 4, 5, 3, 6, 9],
# [6, 0, 6, 5, 4, 0, 8],
# [2, 9, 9, 1, 0, 2, 6],
# [1, 9, 10, 9, 10, 3, 8],
# [1, 6, 7, 6, 1, 4, 0],
# [2, 9, 4, 3, 3, 3, 6],
# [9, 4, 3, 4, 6, 4, 6],
# [2, 7, 4, 1, 2, 7, 5],
# [2, 1, 2, 7, 9, 1, 9],
# [4, 7, 5, 8, 0, 1, 3],
# [6, 10, 4, 7, 2, 9, 10],
# [3, 1, 7, 6, 2, 7, 1],
# [3, 0, 9, 4, 5, 6, 3],
# [8, 10, 4, 6, 4, 10, 7],
# [0, 10, 6, 6, 1, 9, 0],
# [3, 9, 10, 4, 1, 8, 9],
# [1, 0, 0, 2, 7, 5, 6],
# [6, 2, 2, 2, 0, 2, 4],
# [0, 4, 4, 6, 3, 7, 10],
# [3, 6, 10, 7, 4, 7, 1],
# [8, 10, 10, 4, 10, 2, 7],
# [4, 9, 4, 7, 2, 3, 5],
# [6, 0, 9, 5, 6, 10, 4],
# [9, 1, 4, 8, 1, 0, 6],
# [1, 1, 2, 0, 9, 7, 3],
# [7, 0, 6, 2, 2, 0, 5],
# [6, 10, 4, 0, 1, 0, 7],
# [1, 7, 6, 3, 8, 6, 6],
# [3, 1, 6, 1, 6, 2, 0],
# [5, 0, 10, 2, 2, 0, 2],
# [1, 8, 6, 0, 3, 5, 6],
# [4, 10, 10, 10, 7, 2, 6],
# [8, 10, 2, 7, 6, 5, 0],
# [8, 7, 3, 1, 4, 8, 3],
# [9, 9, 1, 1, 5, 5, 1],
# [4, 9, 7, 2, 10, 2, 1],
# [7, 8, 4, 10, 3, 9, 0],
# [10, 8, 5, 7, 3, 7, 2],
# [1, 7, 3, 4, 4, 5, 10],
# [10, 5, 4, 1, 3, 0, 0],
# [5, 5, 1, 0, 0, 5, 3],
# [7, 8, 2, 9, 8, 10, 1],
# [8, 7, 1, 6, 4, 4, 4],
# [8, 6, 5, 0, 6, 6, 8],
# [10, 1, 8, 9, 5, 9, 1],
# [5, 9, 1, 1, 10, 8, 0],
# [10, 8, 1, 7, 8, 7, 10],
# [8, 9, 1, 2, 10, 2, 2],
# [6, 0, 8, 0, 6, 8, 2],
# [9, 1, 9, 0, 4, 1, 8],
# [2, 6, 9, 2, 8, 4, 6],
# [3, 2, 1, 5, 0, 4, 9],
# [9, 10, 4, 1, 4, 5, 0],

])

# 计算 Pearson 相关系数的函数
def pearson_similarity(matrix):
    num_users = matrix.shape[0]
    similarity_matrix = np.zeros((num_users, num_users))

    for i in range(num_users):
        for j in range(i + 1, num_users):  # 只计算一半，因为矩阵是对称的
            user_i = matrix[i]
            user_j = matrix[j]

            mean_i = np.mean(user_i)
            mean_j = np.mean(user_j)

            numerator = np.sum((user_i - mean_i) * (user_j - mean_j))
            denominator = np.sqrt(np.sum((user_i - mean_i) ** 2) * np.sum((user_j - mean_j) ** 2))

            if denominator == 0:
                similarity = 0
            else:
                similarity = numerator / denominator

            # 填充到亲密关系矩阵并确保矩阵对称
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # 对结果矩阵中的元素四舍五入到两位小数
    similarity_matrix = np.round(similarity_matrix, 2)

    return similarity_matrix


# 计算并打印亲密关系矩阵
similarity_matrix = pearson_similarity(user_item_matrix)
print(similarity_matrix)

#（任务相关性+u*夏普利值）*similarity

user_competence = np.full((user_item_matrix.shape[0]), 0)  # 初始信誉设为 0



# 读取任务相关性
#text_task_completion_data = pd.read_csv("D:\plus\\任务相关性.csv", header=None, names=['对象', '文本任务得分'])
task_relation_data = pd.read_csv("D:\plus\sha\任务相关性.csv", header=None, names=['对象', '任务相关性'])
task_relation_data['任务相关性'] = task_relation_data['任务相关性'].str.extract('(\d+\.\d+)').astype(float)
print(task_relation_data)

# 读取夏普利值数据
shape_values = pd.read_csv("D:\plus\sha\夏普利值.csv", header=None, names=['对象', '夏普利值'])
shape_values['夏普利值'] = shape_values['夏普利值'].str.extract('(\d+\.\d)').astype(float)

print(shape_values)
task_completion_data = pd.merge(task_relation_data, shape_values, on='对象')

task_completion_data['总完成度'] = task_completion_data['夏普利值'] + task_completion_data['任务相关性']

print(task_completion_data)
# 删除包含NaN值的行
task_completion_data.dropna(inplace=True)
print(f"task_completion_data{task_completion_data}")











# 计算用户信誉值
user_reputation = np.zeros(user_item_matrix.shape[0])
print(f"task_completion_data shape: {task_completion_data.shape}")
print(f"user_item_matrix shape: {user_item_matrix.shape}")

for i in range(min(user_item_matrix.shape[0], len(task_completion_data))):
    for j in range(min(user_item_matrix.shape[0], len(task_completion_data))):
        if i != j:
            if similarity_matrix[i, j] < 0:
                if not pd.isna(task_completion_data.iloc[i]['总完成度']):
                    user_reputation[i] = task_completion_data.iloc[i]['总完成度']
            else:
                if not pd.isna(task_completion_data.iloc[i]['总完成度']) and not pd.isna(
                        task_completion_data.iloc[j]['总完成度']):
                    user_reputation[i] += similarity_matrix[i, j] * task_completion_data.iloc[j]['总完成度']

total_reputation = np.sum(user_reputation)

print(f"total_reputation: {total_reputation}")


# 创建图
G = nx.Graph()

for i in range(user_item_matrix.shape[0]):
  G.add_node(i+1, reputation=user_reputation[i])

for i in range(user_item_matrix.shape[0]):
  for j in range(i+1, user_item_matrix.shape[0]):
      weight = round(similarity_matrix[i, j], 2)
      if weight > 0:
          G.add_edge(i+1, j+1, weight=weight)

# 找到信誉值最高的用户
core_user_index = np.argmax(user_reputation)
core_user_reputation = user_reputation[core_user_index]

# 找到信誉值高于65%的核心用户
condition = 0.65 * core_user_reputation
core_users = [i for i, reputation in enumerate(user_reputation) if reputation >= condition]
if core_user_index not in core_users:
  core_users.append(core_user_index)

print("信誉值高于65%的核心用户:")
for user_index in core_users:
  print(f"用户 {user_index + 1} - 信誉值: {user_reputation[user_index]}")

# 找到与核心用户群相邻且权重大于0.3的其他用户
threshold_weight = 0.3
neighboring_users = []

for user_index, similarities in enumerate(similarity_matrix):
  if user_index not in core_users:
      for core_user_index in core_users:
          if similarities[core_user_index] > threshold_weight:
              neighboring_users.append(user_index)

neighboring_users = list(set(neighboring_users))

print("与核心用户群相邻且权重大于0.3的其他用户:")
for user_index in neighboring_users:
  print(f"用户 {user_index + 1}")

# 设置节点颜色
node_colors = ['yellow' if i not in core_users and i not in neighboring_users else 'blue' if i in neighboring_users else 'red' for i in range(user_item_matrix.shape[0])]

print("统计亲密用户的人数: " + str(len(neighboring_users)))
print("统计核心用户的人数: " + str(len(core_users)))
print("第二层亲密关系用户: " + str(user_item_matrix.shape[0] - len(neighboring_users) - len(core_users)))

# 计算核心用户和亲密用户的信誉值
core_users_reputation = [user_reputation[user_index] for user_index in core_users]
neighboring_users_reputation = [user_reputation[user_index] for user_index in neighboring_users]

print("核心用户的信誉值:")
for user_index, reputation in zip(core_users, core_users_reputation):
  print(f"用户 {user_index + 1} - 信誉值: {reputation}")

print("\n亲密用户的信誉值:")
for user_index, reputation in zip(neighboring_users, neighboring_users_reputation):
  print(f"用户 {user_index + 1} - 信誉值: {reputation}")

other_users_reputation = [user_reputation[user_index] for user_index in range(user_item_matrix.shape[0]) if user_index not in core_users and user_index not in neighboring_users]

print("\n其他用户的信誉值:")
for user_index, reputation in enumerate(other_users_reputation, start=1):
  print(f"用户 {user_index} - 信誉值: {reputation}")

total_other_reputation = sum(other_users_reputation)
print("其他用户的总信誉值:", total_other_reputation)

total_core_reputation = sum(core_users_reputation)
total_neighboring_reputation = sum(neighboring_users_reputation)

print("\n核心用户的总信誉值:", total_core_reputation)
print("亲密用户的总信誉值:", total_neighboring_reputation)
#
# # 绘制图形
# plt.figure(figsize=(10, 8))
# pos = nx.spring_layout(G)
# pos[core_user_index + 1] = np.array([0, 0])
# nx.draw(G, pos, with_labels=True, font_weight='bold', node_color=node_colors)
#
# edge_labels = {(u, v): round(d['weight'], 2) for u, v, d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#
# node_labels = nx.get_node_attributes(G, 'reputation')
# nx.draw_networkx_labels(G, pos, labels=node_labels, verticalalignment='bottom')
#
# plt.show()
# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np
#
# # 假设 G, user_reputation, core_users, neighboring_users 等变量已经定义
#
# # 设置颜色映射
# color_map = plt.cm.YlOrRd  # 使用 YlOrRd 颜色映射
# norm = mcolors.Normalize(vmin=min(user_reputation), vmax=max(user_reputation))
#
# # 设置节点颜色和大小
# node_colors = [color_map(norm(rep)) if i+1 not in core_users and i+1 not in neighboring_users
#              else 'skyblue' if i+1 in neighboring_users
#              else 'lightgreen' for i, rep in enumerate(user_reputation)]
# node_sizes = [300 + 1000 * (rep / max(user_reputation)) for rep in user_reputation]
#
# # 设置边的颜色和宽度
# edge_colors = [color_map(norm(G[u][v]['weight'])) for u, v in G.edges()]
# edge_widths = [0.5 + 2 * G[u][v]['weight'] for u, v in G.edges()]
#
# # 创建图形
# plt.figure(figsize=(16, 12))
# pos = nx.spring_layout(G, k=0.5, iterations=50)
#
# # 绘制边
# nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)
#
# # 绘制节点
# nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
#
# # 添加标签
# labels = {node: f"{node}\n{user_reputation[node-1]:.2f}" for node in G.nodes()}
# nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")
#
# # 添加图例
# plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map),
#            label="Edge Weight / Node Reputation", shrink=0.8)
#
# # 添加标题和说明
# plt.title("User Similarity Network", fontsize=20, fontweight="bold")
# plt.text(0.05, 0.05, "Node color: Green = Core, Blue = Neighboring, Gradient = Others\n"
#                    "Node size: Proportional to user reputation\n"
#                    "Edge color and width: Proportional to similarity",
#        transform=plt.gca().transAxes, fontsize=10, verticalalignment="bottom")
#
# plt.axis('off')
# plt.tight_layout()
# plt.show()
# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np
#
# # 假设 G, user_reputation, core_users, neighboring_users 等变量已经定义
#
# # 设置颜色映射
# color_map = plt.cm.YlOrRd  # 使用 YlOrRd 颜色映射
# norm = mcolors.Normalize(vmin=min(user_reputation), vmax=max(user_reputation))
#
# # 设置节点颜色和大小
# node_colors = [color_map(norm(rep)) if i+1 not in core_users and i+1 not in neighboring_users
#              else 'skyblue' if i+1 in neighboring_users
#              else 'lightgreen' for i, rep in enumerate(user_reputation)]
# node_sizes = [300 + 1000 * (rep / max(user_reputation)) for rep in user_reputation]
#
# # 设置边的颜色和宽度
# edge_colors = ['black' for _ in G.edges()]  # 将所有边的颜色设置为黑色
# edge_widths = [0.5 + 2 * G[u][v]['weight'] for u, v in G.edges()]
#
# # 创建图形
# plt.figure(figsize=(16, 12))
# pos = nx.spring_layout(G, k=0.5, iterations=50)
#
# # 绘制边
# nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)
#
# # 绘制节点
# nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
#
# # 添加标签
# labels = {node: f"{node}\n{user_reputation[node-1]:.2f}" for node in G.nodes()}
# nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")
#
# # 添加图例
# plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map),
#            label="Node Reputation", shrink=0.8)
#
# # 添加标题和说明
# plt.title("User Relationship Network", fontsize=20, fontweight="bold")
# plt.text(0.05, 0.05, "Node color: Green = Core, Blue = Neighboring, Gradient = Others\n"
#                    "Node size: Proportional to user reputation\n"
#                    "Edge width: Proportional to similarity",
#        transform=plt.gca().transAxes, fontsize=10, verticalalignment="bottom")
#
# plt.axis('off')
# plt.tight_layout()
# plt.show()


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# 假设 G, user_reputation, core_users, neighboring_users 等变量已经定义

# 设置颜色映射
color_map = plt.cm.YlOrRd  # 使用 YlOrRd 颜色映射
norm = mcolors.Normalize(vmin=min(user_reputation), vmax=max(user_reputation))

# 设置节点颜色和大小
node_colors = [color_map(norm(rep)) if i+1 not in core_users and i+1 not in neighboring_users
             else 'skyblue' if i+1 in neighboring_users
             else 'lightgreen' for i, rep in enumerate(user_reputation)]
node_sizes = [100 + 500 * (rep / max(user_reputation)) for rep in user_reputation]  # 减小节点大小

# 设置边的颜色和宽度
edge_colors = ['gray' for _ in G.edges()]  # 将边的颜色设置为灰色
edge_widths = [0.1 + G[u][v]['weight'] for u, v in G.edges()]  # 减小边的宽度

# 创建图形
plt.figure(figsize=(20, 15))  # 增大图形尺寸
pos = nx.spring_layout(G, k=0.3, iterations=50)  # 减小 k 值使节点更紧凑

# 绘制边
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

# 添加标签
labels = {node: f"{node}\n{user_reputation[node-1]:.2f}" for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight="bold")  # 减小字体大小

# 添加图例
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map),
           label="Node Reputation", shrink=0.8)

# 添加标题和说明
plt.title("User Relationship Network", fontsize=20, fontweight="bold")
plt.text(0.05, 0.05, "Node color: Green = Core, Blue = Neighboring, Gradient = Others\n"
                   "Node size: Proportional to user reputation\n"
                   "Edge width: Proportional to similarity",
       transform=plt.gca().transAxes, fontsize=10, verticalalignment="bottom")

plt.axis('off')
plt.tight_layout()
plt.show()