# import numpy as np
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
#
# user_item_matrix = np.array([
#     [3, 4, 0, 2, 7, 6, 0],
#     [1, 2, 4, 0, 7, 5, 0],
#     [2, 4, 8, 0, 9, 6, 5],
#     [3, 1, 0, 4, 1, 2, 3],
#     [7, 8, 3, 6, 0, 1, 4],
#     [5, 8, 4, 7, 4, 9, 7]
# ])
#
# avg_ratings = np.mean(user_item_matrix, axis=1)
#
#
# def calculate_intimacy(user1, user2):
#     numerator = np.sum((user1 - avg_ratings[0]) * (user2 - avg_ratings[1]))
#     denominator = np.sqrt(np.sum((user1 - avg_ratings[0])**2) * np.sum((user2 - avg_ratings[1])**2))
#     intimacy = numerator / denominator
#     return intimacy
#
#
# intimacy_matrix = np.zeros((user_item_matrix.shape[0], user_item_matrix.shape[0]))
# for i in range(user_item_matrix.shape[0]):
#     for j in range(user_item_matrix.shape[0]):
#         intimacy_matrix[i, j] = calculate_intimacy(user_item_matrix[i], user_item_matrix[j])
# print(np.around(intimacy_matrix, 2))
#
#
# user_reputation = np.full((user_item_matrix.shape[0]), 50)  # 初始信誉设为 50
#
#
# task_completion_data = pd.read_csv("D:\plus\similarities-main\similarities-main\examples\\任务相关性.csv", header=None, names=['对象', '得分'])
# task_completion_data['得分'] = task_completion_data['得分'].str.extract('(\d+\.\d+)').astype(float)
#
#
# task_completion = dict(zip(task_completion_data['对象'], task_completion_data['得分']))
#
# # 计算用户信誉值
# for i in range(user_item_matrix.shape[0]):
#     for j in range(user_item_matrix.shape[0]):
#         if i != j:
#             if intimacy_matrix[i, j] < 0:  # Check if intimacy is negative
#                 user_reputation[i] = task_completion[f'对象{i+1}']  # Set reputation to task completion
#             else:
#                 user_reputation[i] += intimacy_matrix[i, j] * task_completion[f'对象{j+1}']
#
#
# G = nx.Graph()
#
#
# for i in range(1, user_item_matrix.shape[0] + 1):
#     G.add_node(i, reputation=user_reputation[i-1])
#
# for i in range(user_item_matrix.shape[0]):
#     for j in range(i+1, user_item_matrix.shape[0]):
#         weight = round(intimacy_matrix[i, j], 2)
#         if weight > 0:
#             G.add_edge(i+1, j+1, weight=weight)
# # 绘制图形
# pos = nx.spring_layout(G)  # 定义节点位置
# nx.draw(G, pos, with_labels=True, font_weight='bold')
# node_labels = nx.get_node_attributes(G, 'reputation')
# nx.draw_networkx_labels(G, pos, labels=node_labels, verticalalignment='bottom')  # 调整标签位置
# plt.show()
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
user_item_matrix = np.array([[3, 3, 1, 7, 9, 5, 7],
                             [10, 7, 9, 0, 2, 3, 9],
                             [5, 1, 1, 3, 9, 8, 3],
                             [5, 9, 8, 2, 6, 9, 2],
                             [1, 4, 8, 4, 10, 4, 4],
                             [0, 4, 7, 9, 10, 8, 9],
                             [4, 1, 4, 2, 9, 8, 1],
                             [6, 0, 6, 10, 8, 6, 1],
                             [2, 10, 6, 4, 0, 0, 9],
                             [6, 10, 8, 6, 10, 8, 8],
                             [7, 2, 4, 9, 7, 1, 6],
                             [9, 3, 10, 8, 6, 0, 9],
                            ])

avg_ratings = np.mean(user_item_matrix, axis=1)

def calculate_intimacy(user1, user2):
    numerator = np.sum((user1 - avg_ratings[0]) * (user2 - avg_ratings[1]))
    denominator = np.sqrt(np.sum((user1 - avg_ratings[0])**2) * np.sum((user2 - avg_ratings[1])**2))
    intimacy = numerator / denominator
    return intimacy

intimacy_matrix = np.zeros((user_item_matrix.shape[0], user_item_matrix.shape[0]))
for i in range(user_item_matrix.shape[0]):
    for j in range(user_item_matrix.shape[0]):
        intimacy_matrix[i, j] = calculate_intimacy(user_item_matrix[i], user_item_matrix[j])

user_reputation = np.full((user_item_matrix.shape[0]), 50)  # 初始信誉设为 50



print(f"intimacy_matrix{intimacy_matrix}")
# 从第一个文件中读取图片clipmult数据
#image_task_completion_data = pd.read_csv("D:\plus\similarities-main\similarities-main\examples\\任务相关性.csv", header=None, names=['对象', '图片任务得分'])
image_task_completion_data = pd.read_csv("D:\plus\\random_data_image.csv", header=None, names=['对象', '图片任务得分'])
image_task_completion_data['图片任务得分'] = image_task_completion_data['图片任务得分'].str.extract('(\d+\.\d+)').astype(float)

# 从第二个文件中读取文本clipmult数据
#text_task_completion_data = pd.read_csv("D:\plus\\任务相关性.csv", header=None, names=['对象', '文本任务得分'])
text_task_completion_data = pd.read_csv("D:\plus\\random_data_text.csv", header=None, names=['对象', '文本任务得分'])
text_task_completion_data['文本任务得分'] = text_task_completion_data['文本任务得分'].str.extract('(\d+\.\d+)').astype(float)

# 合并clipmult数据
task_completion_data = pd.merge(image_task_completion_data, text_task_completion_data, on='对象')
task_completion_data['总完成度'] = task_completion_data['图片任务得分'] + task_completion_data['文本任务得分']

# 计算用户信誉值
for i in range(user_item_matrix.shape[0]):
    for j in range(user_item_matrix.shape[0]):
        if i != j:
            if intimacy_matrix[i, j] < 0:  # Check if intimacy is negative
                if not np.isnan(task_completion_data.loc[i, '总完成度']):
                    user_reputation[i] = task_completion_data.loc[i, '总完成度']  # Set reputation to total completion
            else:
                if not np.isnan(task_completion_data.loc[i, '总完成度']) and not np.isnan(task_completion_data.loc[j, '总完成度']):
                    user_reputation[i] += intimacy_matrix[i, j] * task_completion_data.loc[j, '总完成度']




#计算每个结点的值
G = nx.Graph()

for i in range(1, user_item_matrix.shape[0] + 1):
    G.add_node(i, reputation=user_reputation[i-1])

for i in range(user_item_matrix.shape[0]):
    for j in range(i+1, user_item_matrix.shape[0]):
        weight = round(intimacy_matrix[i, j], 2)
        if weight > 0:
            G.add_edge(i+1, j+1, weight=weight)


norm = mcolors.Normalize(vmin=min(user_reputation), vmax=max(user_reputation))
cmap = plt.cm.get_cmap('rainbow')

node_colors = [cmap(norm(reputation)) for reputation in user_reputation]



# Find the user with the highest reputation
core_user_index = np.argmax(user_reputation)
core_user_reputation = user_reputation[core_user_index]

threshold=0.5
# 假设您有一个亲密度矩阵'intimacy_matrix'，表示用户之间的相似度
# 找到与核心用户U1相似度较高的用户
similar_users = [i for i, sim in enumerate(intimacy_matrix[core_user_index]) if sim > threshold]

# # 确保当节点数量少于30时，用户集占据数据集的1/4
# if len(similar_users) < 30:
#     similar_users = similar_users[:len(similar_users) // 4]

# 设置节点颜色
node_colors = ['yellow' if i != core_user_index and i not in similar_users else 'blue' if i in similar_users else 'red' for i in range(user_item_matrix.shape[0])]

# 将核心用户的节点颜色设置为红色
node_colors[core_user_index] = 'red'
#首先通过核心用户，然后利用核心用户与其他用户之间的关系来发布任务和选择用户。
# 将相似用户的节点颜色设置为蓝色
for user_index in similar_users:
    if user_index != core_user_index:
        node_colors[user_index] = 'blue'
#基于声誉相似度网络，选择用户来满足任务对声誉价值的要求，并选择较高信誉的用户来完成任务，从而提高任务的完成率和质量。
# 绘制图形并设置节点颜色
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # 定义节点位置

# 将核心用户的位置设置为图的中心
pos[core_user_index + 1] = np.array([0, 0])
nx.draw(G, pos, with_labels=True, font_weight='bold', node_color=node_colors)

edge_labels = {(u, v): round(d['weight'], 2) for u, v, d in G.edges(data=True)}

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
node_labels = nx.get_node_attributes(G, 'reputation')
nx.draw_networkx_labels(G, pos, labels=node_labels, verticalalignment='bottom')  # 调整标签位置

plt.show()

# # 绘制图形并设置节点颜色
# plt.figure(figsize=(10, 8))
# pos = nx.spring_layout(G)  # 定义节点位置
# nx.draw(G, pos, with_labels=True, font_weight='bold', node_color=node_colors)
# edge_labels = {(u, v): round(d['weight'], 2) for u, v, d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#
# node_labels = nx.get_node_attributes(G, 'reputation')
# nx.draw_networkx_labels(G, pos, labels=node_labels, verticalalignment='bottom')  # 调整标签位置
#
# plt.show()




# # 绘制图形
# pos = nx.spring_layout(G)  # 定义节点位置
# nx.draw(G, pos, with_labels=True, font_weight='bold')
# edge_labels = {(u, v): round(d['weight'], 2) for u, v, d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#
# node_labels = nx.get_node_attributes(G, 'reputation')
# nx.draw_networkx_labels(G, pos, labels=node_labels, verticalalignment='bottom')  # 调整标签位置
#
# plt.show()