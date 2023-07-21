import numpy as np
import pandas as pd
data=pd.read_csv(r'C:\Users\mehme\OneDrive\Masaüstü\DS\PYTHON\ass4\HW4data.csv')
x=data["X"]
y=data["y"]


def standardize(X):
    return (X - X.mean())/X.std(), X.mean(), X.std()



def predict(x,beta0,beta1):
    y = 1/(1 + np.exp(-beta0-beta1*x ))
    return y
def error(beta0,beta1,x,y,Lambda=0.1):
    y_pred = predict(x,beta0,beta1) 
    J = (-1/len(y))*(np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))) + (Lambda /(2*len(y)))*(beta0**2 + beta1**2)
    return y_pred,J


def gradient(y,y_pred,x,beta0,beta1):
    Lambda=0.1
    grad_beta0=(1/len(y)) * np.sum((y_pred - y) * x) + (Lambda/len(y)) * beta0
    grad_beta1=(1/len(y)) * np.sum((y_pred - y) * x * x) + (Lambda/len(y)) * beta1
    return grad_beta0,grad_beta1


                                                                                                              
def update_param(beta0,beta1,grad_beta0,grad_beta1,alpha):
    beta0_new = beta0 - alpha * grad_beta0
    beta1_new = beta1 - alpha * grad_beta1
    return beta0_new,beta1_new


def gradient_descent(data,num_iter,alpha,Lambda,random_seed):
    X=np.array(data['X'])
    y=np.array(data['y'])
    X,muX,sdX = standardize(X)
    
    np.random.seed(random_seed)
    beta0 = np.random.rand()
    beta1 = np.random.rand()
    J_list = []
    for i in range(num_iter):
        y_pred,J = error(beta0,beta1,x,y)
        grad_beta0,grad_beta1 = gradient(y,y_pred,x,beta0,beta1)
        beta0,beta1 = update_param(beta0,beta1,grad_beta0,grad_beta1,alpha)
        J_list.append(J)
        
    
    return J_list,beta0,beta1
print(gradient_descent(data,num_iter = 1000, alpha = 0.01, Lambda=0.1,random_seed=42))
