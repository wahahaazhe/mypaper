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


#任务总预算
budget=[1000,2000,3000,4000,5000,6000,7000]
#用户完成任务的成本
cost=200


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
# 删除包含NaN值的行
task_completion_data.dropna(inplace=True)
print(f"ask_completion_data{task_completion_data}")

# # 查看对象1的任务完成情况
# object1_data = task_completion_data[task_completion_data['对象'] == '对象1']
# print(object1_data)

# 根据user_item_matrix每行的参与者数量计算完成百分比
for i in range(1,user_item_matrix.shape[0]):
    completion_increase = (user_item_matrix[i].sum() // 5) * 0.05  # 每增加5个参与者，完成度增加5%
    # 根据completion_increase更新当前行的clipmult
    if not np.isnan(task_completion_data.loc[i, '总完成度']):
        task_completion_data.loc[i, '总完成度'] += completion_increase

# 打印更新后的任务完成数据
print(task_completion_data)


# 计算用户信誉值
for i in range(1,user_item_matrix.shape[0]):
    for j in range(1,user_item_matrix.shape[0]):
        if i != j:
            if intimacy_matrix[i, j] < 0:  # Check if intimacy is negative
                if not np.isnan(task_completion_data.loc[i, '总完成度']):
                    user_reputation[i] = task_completion_data.loc[i, '总完成度']  # Set reputation to total completion
            else:
                if not np.isnan(task_completion_data.loc[i, '总完成度']) and not np.isnan(task_completion_data.loc[j, '总完成度']):
                    user_reputation[i] += intimacy_matrix[i, j] * task_completion_data.loc[j, '总完成度']
total_reputation = np.sum(user_reputation)

print(f"total_reputation{total_reputation}")
user_reputation = np.array(user_reputation)
total_reputation = float(total_reputation)  # 确保total_reputation是浮点数
budget = float(budget)  # 确保budget是浮点数

payments = np.zeros(len(user_reputation))

# 计算每个用户的奖励
for i in range(len(user_reputation)):
    payments[i] = user_reputation[i] / total_reputation * budget
for user, payment in payments.items():
    print(f"User {user+1} Payment: {payment} utility:{payment-cost}")
#计算每个结点的值


data_quality = user_reputation.copy()  # 复制用户信誉值作为初始数据质量

for i in range(1,user_item_matrix.shape[0]):
    data_quality[i] += task_completion_data.loc[i, "总完成度"]

# 输出每个用户的数据质量
for user, quality in enumerate(data_quality):
    print(f"User{user+1} Data Quality:{quality}")


# 根据规则计算用户的clipmult
for i in range(1, user_item_matrix.shape[0]):
    for j in range(1, user_item_matrix.shape[0]):
        if i != j:
            completion_increase = 0
            # 根据规则增加clipmult
            # 预算每次翻倍时用户任务完成程度提高15%
            if budget[i] >= 2 * budget[j]:
                completion_increase += 0.15
            # 参与人物的人数越多，用户clipmult提高5%
            if j > 5:
                completion_increase += 0.05
            # 更新用户clipmult
            if not np.isnan(task_completion_data.loc[i, '总完成度']):
                task_completion_data.loc[i, '总完成度'] += completion_increase
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


for b in budget:
    payments = {}
    for i in range(len(user_reputation)):
        payments[i] = user_reputation[i] / total_reputation * b

    for user, payment in payments.items():
        print(f"User {user + 1} Payment for Budget {b}: {payment}")

    data_quality = user_reputation.copy()

    for i in range(1, user_item_matrix.shape[0]):
        data_quality[i] += task_completion_data.loc[i, "总完成度"]

    for user, quality in enumerate(data_quality):
        print(f"User {user + 1} Data Quality for Budget {b}: {quality}")

    for i in range(1, user_item_matrix.shape[0]):
        for j in range(1, user_item_matrix.shape[0]):
            if i != j:
                completion_increase = 0
                if b >= 2 * budget[j]:
                    completion_increase += 0.15
                if j > 5:
                    completion_increase += 0.05
                if not np.isnan(task_completion_data.loc[i, '总完成度']):
                    task_completion_data.loc[i, '总完成度'] += completion_increase

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

    # 绘制以人数为横坐标的信誉值折线图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(user_reputation) + 1), user_reputation, marker='o', color='orange', label='Reputation')
plt.xlabel('User')
plt.ylabel('Reputation')
plt.title(f'Reputation vs. User')
plt.grid(True)
plt.legend()
plt.show()

# 绘制以预算为横坐标的信誉值折线图
plt.figure(figsize=(10, 6))
plt.plot(budget, user_reputation, marker='o', color='green', label='Reputation')
plt.xlabel('Budget')
plt.ylabel('Reputation')
plt.title(f'Reputation vs. Budget')
plt.grid(True)
plt.legend()
plt.show()
