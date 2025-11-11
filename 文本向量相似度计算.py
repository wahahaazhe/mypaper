import csv

from similarities import BertSimilarity
import os
m = BertSimilarity(model_name_or_path="shibing624/text2vec-base-chinese")
task='关于今天天气的描述'


def read_text_files(folder_path,path_prefix):
    files = os.listdir(folder_path)
    text_files = [os.path.join(path_prefix,file) for file in files if file.endswith(('.txt'))]
    print(text_files)
    return text_files
if __name__ == '__main__':
    r=[None]*7
    person=[None]*7
    duixiang=[]
    defen=[]
    i=1
    folder_path="D:\plus\\text_data"
    path_prefix=folder_path
    text_files=read_text_files(path_prefix,folder_path)
    for file in text_files:
        duixiang.append("对象"+str(i))
        similarity_score=m.similarity(task,file).item()
        r[i]=similarity_score*100
        print(f"similarity score: {r[i]:.2f}")
        defen.append(str(r[i]))
        i+=1
    with open('result.csv','w',encoding='utf-8')as f:
        fieldnames=['对象','得分']
        writer =csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        for duixiang,defen in zip(duixiang,defen):
            #迭代器来逐个访问每个人物和对应的得分
            writer.writerow({"对象":duixiang,'得分':defen})


    #
# with open('1.txt','r',encoding='utf-8')as f:
#     answer=f.read()
# r = m.similarity(task,answer)
# print(f"similarity score: {float(r)}")  # similarity score: 0.855146050453186