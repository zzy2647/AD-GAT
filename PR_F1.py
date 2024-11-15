import os
# from numpy import *
import numpy as np

# 计算F1分数
def F1_c(p,r):

    f1 = (2* p* r)/(p+r)

    return f1

# 设置类别 初始化保存数据的矩阵
class_name = ['AD', 'MCI', 'NC']
num_class = len(class_name)
F_matrix=np.mat(np.zeros((num_class,num_class)))

# 混淆矩阵文件路径
BasePath = './checkpoint/TResnetM/'

#打开文件，保存数据至F_matrix矩阵
with open(os.path.join(BasePath, 'new_conf_matrix.txt'), 'r') as f:
    for i,line in enumerate(f):
        if (i<3):
            # print(line.strip().split('\t\t'))
            split_data = line.strip().split('\t\t')
            # split_data = line.strip().split('  ')

            for j in range(3):
                F_matrix[i,j] = split_data[j]

print(F_matrix) 
F1_all = 0
#输出每个类的准确的、召回率、F1
for i in range(num_class):
    print(class_name[i]+'类别的各参数')
    rowsum, colsum = np.sum(F_matrix[i]), sum(F_matrix[r,i] for r in range(num_class))

    precision = F_matrix[i,i]/float(colsum)
    recall = F_matrix[i,i]/float(rowsum)
    F1_SCORE = F1_c(precision,recall)
    F1_all = F1_all + F1_SCORE

    print(f"准确率:{precision:.4f},"+f"召回率:{recall:.4f},"+f"F1:{F1_SCORE:.4f},")
print(f"F1_Macro:{F1_all/3.0: .4f}")

    




