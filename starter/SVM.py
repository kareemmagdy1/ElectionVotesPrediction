import numpy as np
import matplotlib.pyplot as plt


lam=0.01
alpha=0.01
iter=1000
w=None
b=None


def fit(x,y):
    y_predict=np.where(y<=0,-1,1)
    n_samples,n_features=x.shape

    w=np.zeros(n_features)
    b=0

    for _ in range(iter):
        for index,x_i in enumerate(x):
            condition=y_predict[index]*(np.dot(x_i,w)-b)>=1
            if(condition):
                w-=alpha*(2*lam*w)
            else:
                w-=alpha*(2*lam*w-np.dot(x_i,y_predict[index]))
                b-=alpha*y_predict[index]


def predict(x):
    output=np.dot(x,w)-b
    return np.sign(output)
