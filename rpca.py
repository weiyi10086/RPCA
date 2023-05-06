import math
import numpy.linalg
import sys
import numpy as np
from scipy.sparse.linalg import svds
# 引入ElementTree
import xml.etree.ElementTree as et
# 导入random包
import random
import os
import time

def robust_pca(M):
    """ 
    Decompose a matrix into low rank and sparse components.
    将矩阵分解为低秩和稀疏分量。
    Computes the RPCA decomposition using Alternating Lagrangian Multipliers.
    使用交替拉格朗日乘数计算 RPCA 分解。
    Returns L,S the low rank and sparse components respectively
    分别返回 L,S 的低秩和稀疏分量
    """
    L = numpy.zeros(M.shape)
    S = numpy.zeros(M.shape)
    Y = numpy.zeros(M.shape)
    # print M.shape
    mu = (M.shape[0] * M.shape[1]) / (4.0 * L1Norm(M))
    lamb = max(M.shape) ** -0.5
    while not converged(M,L,S):
        L = svd_shrink(M - S - (mu**-1) * Y, mu)
        S = shrink(M - L + (mu**-1) * Y, lamb * mu)
        Y = Y + mu * (M - L - S)
    return L,S
    
def svd_shrink(X, tau):
    """
    Apply the shrinkage operator to the singular values obtained from the SVD of X.
    将收缩算子应用于从 X 的 SVD 获得的奇异值。
    The parameter tau is used as the scaling parameter to the shrink function.
    参数 tau 用作收缩函数的缩放参数。
    Returns the matrix obtained by computing U * shrink(s) * V where
    返回通过计算 U * shrink(s) * V 获得的矩阵，其中
        U are the left singular vectors of X
        U 是 X 的左奇异向量
        V are the right singular vectors of X
        V 是 X 的右奇异向量
        s are the singular values as a diagonal matrix
        s 是作为对角矩阵的奇异值
    """
    U,s,V = numpy.linalg.svd(X, full_matrices=False)
    return numpy.dot(U, numpy.dot(numpy.diag(shrink(s, tau)), V))
    
def shrink(X, tau):
    """
    Apply the shrinkage operator the the elements of X.
    对 X 的元素应用收缩运算符。
    Returns V such that V[i,j] = max(abs(X[i,j]) - tau,0).
    返回满足 V[i,j] = max(abs(X[i,j]) - tau,0) 的 V。
    """
    V = numpy.copy(X).reshape(X.size)
    for i in range(V.size):
        V[i] = math.copysign(max(abs(V[i]) - tau, 0), V[i])
        if V[i] == -0:
            V[i] = 0
    return V.reshape(X.shape)
            
def frobeniusNorm(X):
    """
    Evaluate the Frobenius norm of X
    评估 X 的 Frobenius 范数
    Returns sqrt(sum_i sum_j X[i,j] ^ 2)
    返回 sqrt(sum_i sum_j X[i,j] ^ 2)
    """
    accum = 0
    V = numpy.reshape(X,X.size)
    for i in range(V.size):
        accum += abs(V[i] ** 2)
    return math.sqrt(accum)

def L1Norm(X):
    """
    Evaluate the L1 norm of X
    评估 X 的 L1 范数
    Returns the max over the sum of each column of X
    返回 X 的每列总和的最大值
    """
    return max(numpy.sum(X,axis=0))

def converged(M,L,S):
    """
    A simple test of convergence based on accuracy of matrix reconstruction
    from sparse and low rank parts
    基于矩阵重构精度的简单收敛测试来自稀疏和低秩部分
    """
    error = frobeniusNorm(M - L - S) / frobeniusNorm(M)
    # print "error =", error
    return error <= 10e-6

def readxml(filename,number):
    # 读入xml
    tree = et.ElementTree(file=filename)
    # tree = et.ElementTree(file="./IntraTM-2005-01-01-01-00.xml")
    # 获取根节点
    root = tree.getroot()
    # 生成一个空矩阵
    graph = np.zeros((23, 23))

    # #循环获取数据，生成为一个矩阵
    for tag in root:
        for a_tag in tag:
            if len(a_tag.attrib) != 0:
                source_node = int(a_tag.attrib['id'])
            else:
                continue
            for b_tag in a_tag:
                target_node = int(b_tag.attrib['id'])
                weight = float(b_tag.text)
                graph[source_node - 1][target_node - 1] = weight
    # 获取矩阵的最大和最小值，归一化矩阵
    max = np.max(graph)
    min = np.min(graph)
    m,n = graph.shape
    for i in range(m):
        for j in range(n):
            graph[i,j] = (graph[i,j]-min)/(max-min)

    # 生成噪声
    # 噪声数量
    x = generateGuass(number)
    i,j = generateLocal(number,m)
    graph[i,j] = x

    L, S = robust_pca(graph)
    Y = S[i,j]
    n = np.count_nonzero(Y)
    return n

def write_file(result,n):
    filename = "RPCA" + str(n) + "False.txt"
    file = open(filename, "w+")
    sum=0
    length = len(result)
    for i in result:
        sum+=i
        file.write(str(i)+'\n')
    aver = sum / (length*n)
    file.write('平均为: '+str(aver))
    file.close()



# 随机生成插入异常值的位置
# num:异常值数量
# m:矩阵行列数
def generateLocal(num,m):
    # 生成num个[1,m)的随机整数
    i = random.sample(range(0, m), num)
    j = random.sample(range(0, m), num)
    return i,j

# 生成噪声
# num:生成噪声数量
def generateGuass(num):
    x = []
    # 循环生成高斯噪声
    for i in range(num):
        y = np.random.normal(loc=2.0, scale=2.0, size=None)
        x.append(y)
    return x

# 获取文件路径
def getFileName(number):
    # 要获取文件名的文件夹路径
    folder_path = "./traffic-matrices"
    # 使用os.listdir()函数获取文件夹下的所有文件名
    file_names = os.listdir(folder_path)
    # 打印所有文件名
    result = []
    for file_name in file_names:
        file_name=folder_path+"/"+file_name
        n = readxml(file_name,number)

        result.append(n)
        print(file_name)
    result = np.array(result)
    write_file(result,number)

def main():
    x = [1,5,15,20]
    for i in range(4):
        start = time.perf_counter()
        n = x[i]
        getFileName(n)
        end = time.perf_counter()
        filename = "RPCA" + str(n) + "False.txt"
        print(filename)
        file = open(filename, "a")
        file.write('程序运行时间为: %s Seconds' % (end - start))
        file.close()

if __name__ == '__main__':
    main()