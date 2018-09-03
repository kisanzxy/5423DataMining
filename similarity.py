import numpy as np


"""
calculate SMC of teo sequences
    param:  x: list of 0/1
            y: list of 0/1
    ret: a float value
"""
def SMC(x,y):
    x = np.mat(x)
    y = np.mat(y)
    N, M = np.shape(x)
    sim = ((x*y.T)+((1-x)*(1-y).T))/M
    return sim

def JAC(x,y):
    x = np.mat(x)
    y = np.mat(y)
    N, M = np.shape(x)
    sim = (x*y.T)/(M-(1-x)*(1-y).T)
    return sim
    
def COS(x,y):
    x = np.mat(x)
    y = np.mat(y)
    N, M = np.shape(x)
    sim = (x*y.T)/(np.sqrt(sum(np.power(x.T,2)))*np.sqrt(sum(np.power(y.T,2))))
    return sim

def HammDist(x,y):
    count = 0
    for i in zip(x,y):
        if i[0] != i[1]:
            count+=1
    return count

def Manhattan(x,y):
    d = 0
    for i in range(len(x)):
        d += abs(x[i]-y[i])
    return d

def Euclidean(x,y):
    x = np.mat(x)
    y = np.mat(y)
    return np.linalg.norm(x - y)
