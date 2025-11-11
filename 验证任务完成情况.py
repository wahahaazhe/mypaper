# -*- coding: utf-8 -*-
"""
@author:wangwei
@description:
"""
import glob
import sys
import os
from PIL import Image
import csv
sys.path.append('..')
from similarities import ImageHashSimilarity, SiftSimilarity, ClipSimilarity

#
# def sim_and_search(m):
#     print(m)
#     # similarity
#     sim_scores = m.similarity(imgs1, imgs2)
#     print('sim scores: ', sim_scores)
#     for (idx, i), j in zip(enumerate(image_fps1), image_fps2):
#         s = sim_scores[idx] if isinstance(sim_scores, list) else sim_scores[idx][idx]
#         print(f"{i} vs {j}, score: {s:.4f}")
#     # search
#     m.add_corpus(corpus_imgs)
#     queries = imgs1
#     res = m.most_similar(queries, topn=3)
#     print('sim search: ', res)
#     for q_id, c in res.items():
#         print('query:', image_fps1[q_id])
#         print("search top 3:")
#         for corpus_id, s in c.items():
#             print(f'\t{m.corpus[corpus_id].filename}: {s:.4f}')
#     print('-' * 50 + '\n')

# 定义一个函数来读取文件夹中的所有图片文件
def read_image_files(folder_path,path_prefix):
    files = os.listdir(folder_path)
    image_files = [os.path.join(path_prefix,file) for file in files if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    # os.path.join(path_prefix, file)是Python中os.path
    # 模块提供的一个函数，用于将多个路径组合成一个完整的路径。在这个函数中，path_prefix
    # 是要添加的路径前缀，file是文件名或路径的一部分。


    print(image_files)
    return image_files


# # 调用函数读取两个文件夹中的图片文件
# folder_path1 = 'path_to_first_folder'
# folder_path2 = 'path_to_second_folder'
# read_images_from_folders(folder_path1, folder_path2)


def clip_demo(n1):
    m = ClipSimilarity(model_name_or_path="OFA-Sys/chinese-clip-vit-base-patch16")
    # english model name: openai/clip-vit-base-patch32
    print(m)
    # similarity score between text and image
    #former
    # image_fps = [
    #     'data/image3.png',  # yellow flower image
    #     'data/image1.png',  # tiger image
    #     'data/image5.png',#person
    #     'data/image6.png'#dog
    # ]
    #nowadays
    image_fps=n1
    total_score=0
    tast_des = ['a yellow flower', '一只睡觉的老虎', '一头狮子', '一个人','四只狗']
    imgs = [Image.open(i) for i in image_fps]
    #图片与任务描述进行匹配
    sim_scores = m.similarity(imgs, tast_des)

    best_matches = {}
    #图片文本进行对应
    for idx, i in enumerate(image_fps):
        best_score = 0
        best_text = ''
        for idy, j in enumerate(tast_des):
            s = sim_scores[idx][idy]
            if s > best_score:
                best_score = s
                best_text = j
        best_matches[i] = best_text
        total_score=total_score+best_score               #总得分
        print(f"Best match for {i}: {best_text}, score: {best_score:.4f}")

    print('-' * 50 + '\n')
    print("总得分"+str(total_score*100))
    defen.append(str(total_score*100))
    #对象1和总得分
    # tmp=0.0
    # b=-1
    # print('sim scores: ', sim_scores)
    # for idx, i in enumerate(image_fps):
    #     for idy, j in enumerate(texts):
    #         s = sim_scores[idx][idy]
    #         print(f"{i} vs {j}, score: {s:.4f}")
    #
    #
    #         # if s>tmp:
    #         #     tmp=s
    #         #     b=b+1
    #         # else:
    #         #     print("maxmatch" + str(i) + " is " +texts[b]  + str(tmp))
    # print('-' * 50 + '\n')


if __name__ == "__main__":
    # folder_path1="D:\plus\similarities-main\similarities-main\examples\data\对象1"
    # folder_path2="D:\plus\similarities-main\similarities-main\examples\data\对象2"
    folder_path=[None]*7
    image_files=[None]*7
    person=[]
    defen=[]
    for i in range(1,7):
        #exec(f'folder_path{i}')
        folder_path[i]="D:\plus\similarities-main\similarities-main\examples\data\对象"+str(i)
        path_prefix=folder_path[i]
        image_files[i]=read_image_files(path_prefix,folder_path[i])
        print('对象'+str(i)+'得分')
        person.append("对象"+str(i))
        clip_demo(image_files[i])
    with open('result.csv','w',encoding='utf-8')as f:
        fieldnames=['对象','得分']
        writer =csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        for person,defen in zip(person,defen):
            #迭代器来逐个访问每个人物和对应的得分
            writer.writerow({"对象":person,'得分':defen})
    # path_prefix = folder_path1
    # image_files1 =read_image_files(path_prefix,folder_path1)
    # path_prefix = folder_path2
    # image_files2 = read_image_files(path_prefix,folder_path2)

    # image_fps1 = ['data/image1.png', 'data/image3.png']
    # image_fps2 = ['data/image12-like-image1.png', 'data/image10.png']
    # imgs1 = [Image.open(i) for i in image_fps1]
    # imgs2 = [Image.open(i) for i in image_fps2]
    # corpus_fps = glob.glob('data/*.jpg') + glob.glob('data/*.png')
    # corpus_imgs = [Image.open(i) for i in corpus_fps]
    # 1. image and text similarity
    # print('对象1得分')
    # clip_demo(image_files1)
    # print('对象2得分')
    # clip_demo(image_files2)
    #
    # # 2. image and image similarity score
    # sim_and_search(ClipSimilarity())  # the best result
    # sim_and_search(ImageHashSimilarity(hash_function='phash'))
    # sim_and_search(SiftSimilarity())
