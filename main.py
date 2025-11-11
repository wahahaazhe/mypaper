import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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





avg_ratings = np.mean(user_item_matrix, axis=1)

def calculate_similary(user1, user2,i,j):
    numerator = np.sum((user1 - avg_ratings[i]) * (user2 - avg_ratings[j]))
    denominator = np.sqrt(np.sum((user1 - avg_ratings[i])**2) * np.sum((user2 - avg_ratings[j])**2))
    intimacy = numerator / denominator
    return intimacy

intimacy_matrix = np.zeros((user_item_matrix.shape[0], user_item_matrix.shape[0]))
print(intimacy_matrix)
for i in range(user_item_matrix.shape[0]):
    for j in range(user_item_matrix.shape[0]):
        intimacy_matrix[i, j] = calculate_similary(user_item_matrix[i], user_item_matrix[j],i,j)

user_reputation = np.full((user_item_matrix.shape[0]), 0)  # 初始信誉设为 0




print(f"intimacy_matrix{intimacy_matrix}")
# 从第一个文件中读取图片clipmult数据
#image_task_completion_data = pd.read_csv("D:\plus\similarities-main\similarities-main\examples\\任务相关性.csv", header=None, names=['对象', '图片任务得分'])
image_task_completion_data = pd.read_csv("D:\plus\\random_data_image.csv", header=None, names=['对象', '图片任务得分'])
image_task_completion_data['图片任务得分'] = image_task_completion_data['图片任务得分'].str.extract('(\d+\.\d+)').astype(float)

# 从第二个文件中读取文本clipmult数据
#text_task_completion_data = pd.read_csv("D:\plus\\任务相关性.csv", header=None, names=['对象', '文本任务得分'])
text_task_completion_data = pd.read_csv("D:\plus\\random_data_text.csv", header=None, names=['对象', '文本任务得分'])
text_task_completion_data['文本任务得分'] = text_task_completion_data['文本任务得分'].str.extract('(\d+\.\d+)').astype(float)
print(text_task_completion_data)
#从第三个文件读取历史完成程度
history_achievement=pd.read_csv("D:\plus\sha\history\history_result.csv",header=None,names=['对象',"历史任务完成程度"])


history_achievement['历史任务完成程度']=history_achievement['历史任务完成程度'].str.extract('(\d+\.\d+)').astype(float)
print(f"history_achievement{history_achievement}")


# 从第四个文件中读取夏普利值数据
shape_values = pd.read_csv("/sha/任务相关性.csv", header=None, names=['对象', '夏普利值'])
shape_values['夏普利值'] = shape_values['夏普利值'].str.extract('(\d+\.\d)').astype(float)

print(shape_values)

# # 获取总共有多少行用户
# num_users = user_item_matrix.shape[0]
#
# # 将shape_values['对象']转换为字符串类型
# shape_values['对象'] = shape_values['对象'].astype(str)
#
# # 创建一个新的Series，索引为shape_values['对象']，初始值为0
# sharp_values_series = pd.Series(0, index=shape_values['对象'])
#
# # 用原始数据中的夏普利值数据填充新的Series
# sharp_values_series.loc[shape_values['对象']] = shape_values['夏普利值']
#
# # 输出结果
# print(f"sharpvalue{sharp_values_series}")

task_completion_data = pd.merge(image_task_completion_data, text_task_completion_data, on='对象')
task_completion_data = pd.merge(task_completion_data, history_achievement, on='对象')
task_completion_data = pd.merge(task_completion_data, shape_values, on='对象')

task_completion_data['总完成度'] = task_completion_data['图片任务得分'] + task_completion_data['文本任务得分']+shape_values['夏普利值']+history_achievement['历史任务完成程度']
# 删除包含NaN值的行
task_completion_data.dropna(inplace=True)
print(f"task_completion_data{task_completion_data}")
# # 查看对象1的任务完成情况
# object1_data = task_completion_data[task_completion_data['对象'] == '对象1']
# print(object1_data)

# 计算用户信誉值
for i in range(1,user_item_matrix.shape[0]):
    for j in range(1,user_item_matrix.shape[0]):
        if i != j:
            if intimacy_matrix[i, j] < 0:  # Check if intimacy is negative
                if not np.isnan(task_completion_data.loc[i, '总完成度']):
                    user_reputation[i] = task_completion_data.loc[i, '总完成度']#+float(sharp_values_series.loc[i])+float(history_achievement.loc[i])  # Set reputation to total completion
            else:
                if not np.isnan(task_completion_data.loc[i, '总完成度']) and not np.isnan(task_completion_data.loc[j, '总完成度']):
                    user_reputation[i] += intimacy_matrix[i, j] * task_completion_data.loc[j, '总完成度']#+float(sharp_values_series.loc[i])+float(history_achievement.loc[i])
total_reputation = np.sum(user_reputation)

print(f"total_reputation{total_reputation}")
payments={}


#计算每个结点的值


data_quality = user_reputation.copy()  # 复制用户信誉值作为初始数据质量

for i in range(1,user_item_matrix.shape[0]):
    data_quality[i] += task_completion_data.loc[i, "总完成度"]

# 输出每个用户的数据质量
for user, quality in enumerate(data_quality):
    print(f"User{user+1} Data Quality:{quality}")


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

#defend coreuse
condition=0.65*core_user_reputation
core_users=[i for i, reputation in enumerate(user_reputation) if reputation >= condition]
# 确保核心用户也包含在类似用户列表中
if core_user_index not in core_users:
    core_users.append(core_user_index)

# 输出信誉值高于65%的核心用户
print("信誉值高于65%的核心用户:")
for user_index in core_users:
    print(f"用户 {user_index + 1} - 信誉值: {user_reputation[user_index]}")

# 假设 intimacy_matrix 是一个表示用户间相似度的矩阵，其中每行对应一个用户与其他用户的相似度
# 假设 core_users 是一个包含核心用户群索引的列表

# 找到权重大于0.3与核心用户群相邻的其他用户
threshold_weight = 0.3
neighboring_users = []

for user_index, similarities in enumerate(intimacy_matrix):
    if user_index not in core_users:
        for core_user_index in core_users:
            if similarities[core_user_index] > threshold_weight:
                neighboring_users.append(user_index)

neighboring_users = list(set(neighboring_users))  # 去除重复的用户索引

# 输出与核心用户群相邻且权重大于0.3的其他用户
print("与核心用户群相邻且权重大于0.3的其他用户:")
for user_index in neighboring_users:
    print(f"用户 {user_index + 1}")



#
# # 计算核心用户群的平均相似度
# avg_core = np.mean(intimacy_matrix[core_users], axis=0)
#
#
# threshold=0.30
# # 假设您有一个亲密度矩阵'intimacy_matrix'，表示用户之间的相似度
# # 找到与核心用户U1相似度较高的用户
# similar_users = [i for i, sim in enumerate(avg_core) if sim > threshold and i not in core_users]

# # 确保当节点数量少于30时，用户集占据数据集的1/4
# if len(similar_users) < 30:
#     similar_users = similar_users[:len(similar_users) // 4]

# 设置节点颜色
node_colors = ['yellow' if i != core_users and i not in neighboring_users else 'blue' if i in neighboring_users else 'red' for i in range(user_item_matrix.shape[0])]

#首先通过核心用户，然后利用核心用户与其他用户之间的关系来发布任务和选择用户。
# 将相似用户的节点颜色设置为蓝色
for user_index in neighboring_users:
    if user_index != core_users:
        node_colors[user_index] = 'blue'

#统计亲密用户的人数，与核心用户相邻
print("统计亲密用户的人数"+str(len(neighboring_users)))

for user_index in core_users:
# 将核心用户的节点颜色设置为红色
    node_colors[user_index] = 'red'

#统计核心用户的人数
print("统计核心用户的人数"+str(len(core_users)))

#统计第二层亲密关系用户的人数
print("第二层亲密关系用户"+str(user_item_matrix.shape[0]-len(neighboring_users)-len(core_users)))

#所选感知用户的信誉
# 计算核心用户和亲密用户的信誉值
core_users_reputation = [user_reputation[user_index] for user_index in core_users]
neighboring_users_reputation = [user_reputation[user_index] for user_index in neighboring_users]

print("核心用户的信誉值:")
for user_index, reputation in zip(core_users, core_users_reputation):
    print(f"用户 {user_index + 1} - 信誉值: {reputation}")

print("\n亲密用户的信誉值:")
for user_index, reputation in zip(neighboring_users, neighboring_users_reputation):
    print(f"用户 {user_index + 1} - 信誉值: {reputation}")

# 计算除了核心用户和亲密用户外的其他用户的信誉值
other_users_reputation = [user_reputation[user_index] for user_index in range(user_item_matrix.shape[0]) if user_index not in core_users and user_index not in neighboring_users]

print("\n其他用户的信誉值:")
for user_index, reputation in enumerate(other_users_reputation, start=1):
    print(f"用户 {user_index} - 信誉值: {reputation}")

# 计算其他用户的总信誉
total_other_reputation = sum(other_users_reputation)
print("其他用户的总信誉值:", total_other_reputation)



# 计算核心用户和亲密用户的总信誉
total_core_reputation = sum(core_users_reputation)
total_neighboring_reputation = sum(neighboring_users_reputation)

print("\n核心用户的总信誉值:", total_core_reputation)
print("亲密用户的总信誉值:", total_neighboring_reputation)


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