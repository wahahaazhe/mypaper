import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 用户-项目矩阵
user_item_matrix = np.array([
    [2, 0, 10, 1, 10, 6, 3],
    [3, 4, 5, 9, 6, 4, 10],
    [0, 2, 7, 6, 9, 4, 3],
    [8, 1, 1, 2, 7, 9, 9],
    [8, 2, 7, 10, 9, 0, 1],
    [7, 1, 0, 4, 0, 2, 1],
    [5, 9, 3, 3, 10, 6, 10],
    [8, 5, 5, 6, 4, 4, 1],
    [7, 4, 2, 5, 1, 0, 3],
    [2, 7, 4, 3, 4, 3, 4],
    [0, 9, 4, 1, 6, 5, 4],
    [6, 0, 10, 9, 2, 5, 8],
    [5, 4, 3, 4, 1, 5, 10],
    [4, 7, 0, 4, 5, 2, 10],
    [1, 10, 3, 10, 9, 1, 8],
    [1, 8, 1, 9, 3, 7, 9],
    [2, 0, 8, 8, 8, 7, 5],
    [4, 2, 7, 1, 0, 6, 9],
    [9, 1, 6, 0, 8, 8, 3],
    [10, 1, 9, 10, 7, 0, 7],
    [8, 2, 5, 7, 0, 3, 7],
    [8, 2, 0, 5, 3, 6, 6],
    [0, 2, 7, 6, 8, 4, 3],
    [7, 5, 6, 2, 0, 3, 5],
    [4, 6, 8, 3, 5, 9, 0],
    [8, 1, 4, 5, 1, 2, 0],
    [3, 7, 4, 4, 4, 5, 1],
    [6, 0, 6, 4, 0, 10, 7],
    [8, 9, 10, 10, 0, 1, 0],
    [0, 7, 2, 7, 5, 5, 3],
    [6, 8, 3, 7, 10, 4, 7],
    [2, 9, 6, 1, 7, 1, 6],
    [9, 3, 1, 9, 2, 1, 0],
    [8, 6, 7, 5, 5, 2, 3],
    [1, 10, 1, 0, 1, 5, 6],
    [6, 5, 1, 6, 6, 7, 7],
    [9, 7, 6, 7, 1, 4, 9],
    [4, 5, 5, 0, 0, 10, 6],
    [7, 7, 6, 1, 7, 6, 10],
    [7, 1, 9, 6, 8, 7, 6],
    [7, 1, 4, 9, 1, 7, 0],
    [4, 7, 2, 2, 10, 10, 1],
    [2, 4, 4, 1, 5, 1, 4],
    [10, 3, 6, 2, 4, 5, 7],
    [1, 1, 2, 4, 7, 10, 9],
    [3, 6, 5, 10, 3, 1, 3],
    [6, 8, 5, 3, 4, 8, 6],
    [4, 4, 1, 0, 9, 2, 1],
    [8, 8, 0, 3, 2, 10, 8],
    [3, 6, 7, 0, 2, 10, 0],
    [9, 0, 3, 8, 6, 2, 4],
    [7, 9, 8, 10, 1, 5, 9],
    [1, 8, 2, 0, 0, 9, 9],
    [6, 4, 4, 9, 0, 2, 8],
    [0, 2, 8, 5, 4, 2, 5],
    [7, 3, 0, 6, 8, 10, 4],
    [5, 8, 4, 1, 3, 8, 7],
    [6, 8, 6, 4, 7, 9, 9],
    [8, 3, 10, 2, 6, 0, 5],
    [4, 0, 2, 5, 4, 0, 7],
    [9, 1, 8, 5, 3, 5, 5],
    [3, 6, 4, 8, 0, 2, 3],
    [5, 2, 8, 6, 10, 4, 5],
    [2, 0, 7, 4, 1, 4, 4],
    [7, 3, 10, 4, 7, 1, 4],
    [0, 1, 5, 3, 5, 3, 8],
    [5, 2, 5, 0, 3, 1, 8],
    [10, 4, 5, 2, 5, 1, 0],
    [0, 6, 2, 7, 2, 1, 7],
    [2, 0, 2, 4, 10, 6, 8],
    [7, 6, 6, 1, 1, 6, 9],
    [5, 0, 6, 8, 2, 7, 4],
    [2, 8, 10, 2, 6, 3, 3],
    [5, 3, 6, 2, 3, 7, 8],
    [5, 7, 7, 8, 3, 9, 3],
    [3, 0, 0, 8, 6, 5, 1],
    [9, 0, 10, 2, 7, 6, 7],
    [1, 8, 1, 10, 2, 7, 4],
    [10, 8, 5, 8, 9, 10, 4],
    [3, 6, 2, 9, 7, 3, 0],
    [9, 8, 2, 4, 5, 5, 10],
    [2, 9, 1, 10, 0, 0, 1],
    [4, 8, 9, 10, 0, 4, 1],
    [4, 9, 9, 2, 0, 10, 9],
    [9, 7, 8, 9, 10, 10, 2],
    [6, 10, 9, 10, 6, 8, 5],
    [6, 7, 6, 4, 1, 7, 9],
    [8, 2, 0, 5, 9, 3, 0],
    [4, 7, 0, 0, 9, 7, 10],
    [9, 2, 1, 7, 7, 2, 8],
    [9, 6, 8, 7, 8, 7, 2],
    [8, 3, 6, 9, 9, 7, 1],
    [6, 8, 1, 2, 8, 10, 8],
    [4, 6, 0, 5, 2, 2, 6],
    [4, 0, 2, 4, 6, 7, 1],
    [4, 6, 1, 8, 8, 3, 3],
    [5, 8, 5, 3, 4, 6, 1],
    [6, 6, 0, 6, 3, 3, 1],
    [0, 5, 8, 5, 3, 0, 3],
    [8, 1, 8, 4, 10, 2, 4],
    [10, 8, 2, 9, 8, 3, 10],
    [0, 3, 4, 6, 5, 8, 3],
    [9, 2, 1, 1, 3, 1, 4],
    [0, 8, 3, 6, 2, 2, 1],
    [10, 3, 7, 4, 1, 2, 6],
    [10, 0, 9, 3, 6, 5, 6],
    [4, 8, 1, 4, 4, 6, 1],
    [5, 0, 8, 6, 8, 1, 5],
    [0, 6, 1, 1, 4, 0, 0],
    [0, 7, 8, 3, 3, 8, 5],
    [5, 7, 7, 9, 9, 5, 1],
    [0, 9, 3, 0, 0, 7, 5],
    [2, 3, 6, 6, 7, 0, 2],
    [2, 2, 3, 10, 1, 1, 2],
    [1, 4, 10, 6, 3, 5, 3],
    [5, 1, 7, 4, 7, 1, 8],
    [8, 7, 7, 8, 10, 7, 10],
    [4, 3, 5, 10, 4, 0, 6],
    [2, 4, 6, 8, 0, 5, 6],
    [5, 0, 10, 9, 4, 8, 7],
    [6, 9, 9, 3, 6, 10, 10],
    [5, 1, 7, 7, 1, 7, 3],
    [3, 4, 3, 6, 4, 4, 5],
    [3, 6, 5, 10, 1, 1, 8],
    [2, 4, 1, 5, 10, 2, 2],
    [6, 10, 7, 1, 10, 3, 8],
    [3, 7, 9, 1, 9, 0, 5],
    [7, 5, 1, 4, 3, 2, 8],
    [0, 4, 1, 1, 10, 10, 10],
    [5, 4, 1, 10, 1, 4, 5],
    [5, 5, 1, 1, 6, 6, 10],
    [4, 7, 10, 2, 3, 6, 10],
    [4, 0, 6, 2, 7, 0, 10],
    [5, 2, 9, 6, 0, 5, 0],
    [5, 5, 2, 8, 3, 8, 8],
    [9, 2, 4, 7, 5, 2, 8],
    [7, 7, 9, 0, 9, 0, 10],
    [4, 1, 7, 2, 5, 0, 0],
    [3, 2, 3, 9, 1, 9, 1],
    [5, 3, 3, 8, 1, 0, 10],
    [8, 3, 3, 6, 6, 9, 1],
    [7, 6, 0, 4, 7, 8, 3],
    [2, 7, 1, 6, 4, 9, 6],
    [2, 0, 1, 4, 3, 2, 8],
    [0, 3, 10, 4, 4, 8, 0],
    [0, 7, 10, 7, 9, 3, 10],
    [1, 7, 8, 6, 4, 6, 10],
    [2, 1, 4, 5, 3, 6, 9],
    [6, 0, 6, 5, 4, 0, 8],
    [2, 9, 9, 1, 0, 2, 6],
    [1, 9, 10, 9, 10, 3, 8],
    [1, 6, 7, 6, 1, 4, 0],
    [2, 9, 4, 3, 3, 3, 6],
    [9, 4, 3, 4, 6, 4, 6],
    [2, 7, 4, 1, 2, 7, 5],
    [2, 1, 2, 7, 9, 1, 9],
    [4, 7, 5, 8, 0, 1, 3],
    [6, 10, 4, 7, 2, 9, 10],
    [3, 1, 7, 6, 2, 7, 1],
    [3, 0, 9, 4, 5, 6, 3],
    [8, 10, 4, 6, 4, 10, 7],
    [0, 10, 6, 6, 1, 9, 0],
    [3, 9, 10, 4, 1, 8, 9],
    [1, 0, 0, 2, 7, 5, 6],
    [6, 2, 2, 2, 0, 2, 4],
    [0, 4, 4, 6, 3, 7, 10],
    [3, 6, 10, 7, 4, 7, 1],
    [8, 10, 10, 4, 10, 2, 7],
    [4, 9, 4, 7, 2, 3, 5],
    [6, 0, 9, 5, 6, 10, 4],
    [9, 1, 4, 8, 1, 0, 6],
    [1, 1, 2, 0, 9, 7, 3],
    [7, 0, 6, 2, 2, 0, 5],
    [6, 10, 4, 0, 1, 0, 7],
    [1, 7, 6, 3, 8, 6, 6],
    [3, 1, 6, 1, 6, 2, 0],
    [5, 0, 10, 2, 2, 0, 2],
    [1, 8, 6, 0, 3, 5, 6],
    [4, 10, 10, 10, 7, 2, 6],
    [8, 10, 2, 7, 6, 5, 0],
    [8, 7, 3, 1, 4, 8, 3],
    [9, 9, 1, 1, 5, 5, 1],
    [4, 9, 7, 2, 10, 2, 1],
    [7, 8, 4, 10, 3, 9, 0],
    [10, 8, 5, 7, 3, 7, 2],
    [1, 7, 3, 4, 4, 5, 10],
    [10, 5, 4, 1, 3, 0, 0],
    [5, 5, 1, 0, 0, 5, 3],
    [7, 8, 2, 9, 8, 10, 1],
    [8, 7, 1, 6, 4, 4, 4],
    [8, 6, 5, 0, 6, 6, 8],
    [10, 1, 8, 9, 5, 9, 1],
    [5, 9, 1, 1, 10, 8, 0],
    [10, 8, 1, 7, 8, 7, 10],
    [8, 9, 1, 2, 10, 2, 2],
    [6, 0, 8, 0, 6, 8, 2],
    [9, 1, 9, 0, 4, 1, 8],
    [2, 6, 9, 2, 8, 4, 6],
    [3, 2, 1, 5, 0, 4, 9],
    [9, 10, 4, 1, 4, 5, 0],

])

# 任务总预算
budget = 1000
cost = 200

# 平均评分
avg_ratings = np.mean(user_item_matrix, axis=1)

def calculate_intimacy(user1, user2):
    numerator = np.sum((user1 - avg_ratings[0]) * (user2 - avg_ratings[1]))
    denominator = np.sqrt(np.sum((user1 - avg_ratings[0])**2) * np.sum((user2 - avg_ratings[1])**2))
    intimacy = numerator / denominator
    return intimacy

# 计算用户间的亲密度矩阵
intimacy_matrix = np.zeros((user_item_matrix.shape[0], user_item_matrix.shape[0]))
for i in range(user_item_matrix.shape[0]):
    for j in range(user_item_matrix.shape[0]):
        intimacy_matrix[i, j] = calculate_intimacy(user_item_matrix[i], user_item_matrix[j])

# 用户初始信誉
user_reputation = np.full((user_item_matrix.shape[0]), 20)

# 读取clipmult数据
image_task_completion_data = pd.read_csv("D:\plus\\random_data_image.csv", header=None, names=['对象', '图片任务得分'])
image_task_completion_data['图片任务得分'] = image_task_completion_data['图片任务得分'].str.extract('(\d+\.\d+)').astype(float)

text_task_completion_data = pd.read_csv("D:\plus\\random_data_text.csv", header=None, names=['对象', '文本任务得分'])
text_task_completion_data['文本任务得分'] = text_task_completion_data['文本任务得分'].str.extract('(\d+\.\d+)').astype(float)

history_achievement = pd.read_csv("D:\plus\sha\history\history_result.csv", header=None, names=['对象',"历史任务完成程度"])
history_achievement['历史任务完成程度'] = history_achievement['历史任务完成程度'].str.extract('(\d+\.\d+)').astype(float)

shape_values = pd.read_csv("/sha/任务相关性.csv", header=None, names=['对象', '夏普利值'])
shape_values['夏普利值'] = shape_values['夏普利值'].str.extract('(\d+\.\d)').astype(float)

task_completion_data = pd.merge(image_task_completion_data, text_task_completion_data, on='对象')
task_completion_data = pd.merge(task_completion_data, history_achievement, on='对象')
task_completion_data = pd.merge(task_completion_data, shape_values, on='对象')
task_completion_data['总完成度'] = task_completion_data['图片任务得分'] + task_completion_data['文本任务得分'] + shape_values['夏普利值'] + history_achievement['历史任务完成程度']
task_completion_data.dropna(inplace=True)

# 计算用户信誉值
for i in range(1, user_item_matrix.shape[0]):
    for j in range(1, user_item_matrix.shape[0]):
        if i != j:
            if intimacy_matrix[i, j] < 0:
                if not np.isnan(task_completion_data.loc[i, '总完成度']):
                    user_reputation[i] = task_completion_data.loc[i, '总完成度']
            else:
                if not np.isnan(task_completion_data.loc[i, '总完成度']) and not np.isnan(task_completion_data.loc[j, '总完成度']):
                    user_reputation[i] += intimacy_matrix[i, j] * task_completion_data.loc[j, '总完成度']

total_reputation = np.sum(user_reputation)
payments = {}
for i in range(len(user_reputation)):
    payments[i] = user_reputation[i] / total_reputation * budget

# 计算数据质量
data_quality = user_reputation.copy()
for i in range(1, user_item_matrix.shape[0]):
    data_quality[i] += task_completion_data.loc[i, "总完成度"]

# 构建信誉网络
G = nx.Graph()
for i in range(1, user_item_matrix.shape[0] + 1):
    G.add_node(i, reputation=user_reputation[i-1])

for i in range(user_item_matrix.shape[0]):
    for j in range(i+1, user_item_matrix.shape[0]):
        weight = round(intimacy_matrix[i, j], 2)
        if weight > 0:
            G.add_edge(i+1, j+1, weight=weight)

# 设置初始条件和步长
start_condition = 0.5
step_size = 0.05
num_steps = int((1.00 - start_condition) / step_size) + 1

for step in range(num_steps):
    condition = start_condition + step * step_size
    print(f"\n当前条件值: {condition}")

    core_users = [i for i, reputation in enumerate(user_reputation) if reputation >= condition * np.max(user_reputation)]
    print("信誉值高于条件的核心用户:")
    for user_index in core_users:
        print(f"用户 {user_index + 1} - 信誉值: {user_reputation[user_index]}")

    neighboring_users = []

    for user_index, similarities in enumerate(intimacy_matrix):
        if user_index not in core_users:
            for core_user_index in core_users:
                if similarities[core_user_index] > 0.3:
                    neighboring_users.append(user_index)

    neighboring_users = list(set(neighboring_users))

    print("与核心用户群相邻且权重大于0.3的其他用户:")
    for user_index in neighboring_users:
        print(f"用户 {user_index + 1}")

    node_colors = ['yellow' if i != core_users and i not in neighboring_users else 'blue' if i in neighboring_users else 'red' for i in range(user_item_matrix.shape[0])]

    for user_index in neighboring_users:
        if user_index != core_users:
            node_colors[user_index]
