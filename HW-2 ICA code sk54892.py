#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import linalg as la
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# In[2]:


''' 
n = number of source signals
m = number of mixed signals
t = length of a signal
X = mixed signal (m,t)
W = initial unmixing matrix (n,m)
'''
def ICA(X, W, learning_rate=0.000001, MAXIMUM_ITERATIONS=1000):

    m,t = X.shape
    n,m = W.shape
    
    print(f'Learning rate={learning_rate}')
    print(f'R_max value is {MAXIMUM_ITERATIONS}')
    
    starting_time = time.time()
    for i in range(0, int(MAXIMUM_ITERATIONS)):
        Y = W@X
        Z = 1/(1 + np.exp(-Y))
        delW = learning_rate*((np.identity(n)*t + (1 - 2*Z)@Y.T) @W)
        
        W += delW
    ending_time = round(time.time() - starting_time, 4)
    print(f'ICA end time {ending_time} sec')
    
    return W


# In[ ]:


def correlation_coefficient(T1, T2):
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result


# In[4]:


def norm(A,spacing):
    Anorm = []
    
    mi = min(map(min, A))
    ma = max(map(max, A))
    
    s = spacing if spacing is not None else 0
    
    for i in range(len(A)):
        Anorm.append(s*i + (A[i]-mi) / (ma-mi))
        
    return Anorm


# In[ ]:


colours = ['m','k','b','g','c']
def plots(A, spacing=None, title=None):   
    
    n,_ = A.shape
    Anorm = norm(A,spacing)
    
    fig = plt.figure(figsize=(8,5))
    ax = plt.gca()
    ax.set_facecolor('#C0C0C0')

    for i in range(0,n):
        plt.plot(Anorm[i][:50], colours[i])
        vis = False if i < n-1 else True
        ax.axes.xaxis.set_visible(vis)
    
    if title: ax.set_title = title
        
    plt.show()


# In[59]:


data = loadmat('sounds.mat')['sounds'] # Load data

np.random.seed(8)

U = data[:3][:]
n,t = U.shape
m = n

A = np.random.rand(m,n) # Mix signals and create W_init
X = A@U

W_init = np.random.rand(n,m)
W = ICA(X, W_init, learning_rate=0.0000001, MAXIMUM_ITERATIONS=1000)

reconstructed_data = W@X


# In[60]:


plots(U,0.05)


# In[61]:


plots(X, 0.05)


# In[62]:


v = np.array([reconstructed_data[1],reconstructed_data[0],reconstructed_data[2]])
plots(reconstructed_data, 0.05)


# In[64]:


c = correlation_coefficient(U,v)
print(round(c,3))


# In[17]:


U = loadmat('sounds.mat')['sounds'] #load data
n,t = U.shape
results = []
eta_trials = [0.00001, 0.000001]
m_trials = np.array([1, 5, 10]) * n
R_max_size = np.array([1000])

for eta in eta_trials:
    for m in m_trials:
        for R_max in R_max_size: 
            np.random.seed(8)
            print(f'm={m}')

            A = np.random.rand(m,n)
            W_init = np.random.rand(n,m)

            X = A@U
            W = ICA(X, W_init, learning_rate=eta, MAXIMUM_ITERATIONS=R_max)

            rec = W@X
            results.append(rec)

            error = la.norm(rec-U,2)
            print(f'Error: {error}\n')


# In[42]:


n = 5
m = 5
U = data[:n,:]

iterations_results = []

iterations_trials = np.array([1000, 10000, 100000])
for R_max in iterations_trials:
    np.random.seed(8)
    
    A = np.random.rand(m,n)
    W_init = np.random.rand(n,m)
    
    X = A@U
    W = ICA(X, W_init, learning_rate=0.000001, MAXIMUM_ITERATIONS=R_max)
    
    reconstructed_data = W@X
    iterations_results.append(reconstructed_data)
    
    errors = la.norm(reconstructed_data-U,2)
    print(f'Error: {errors}\n')


# In[43]:


# m test
np.random.seed(8)
n = 5
U = data[:n,:]
test_results = []
test_errors = []

for m in range(n,21):
    print(f'm={m}')

    A = np.random.rand(m,n)
    W_init = np.random.rand(n,m)

    X = A@U
    W = ICA(X, W_init, learning_rate=0.000001, MAXIMUM_ITERATIONS=1000)

    reconstructed_data = W@X
    test_results.append(rec)

    error = la.norm(reconstructed_data-U,2)
    test_errors.append(error)
    print(f'Error: {error}\n')


# In[45]:


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(4,20), test_errors,  '#800000')
ax.set_xlabel('m')
ax.set_ylabel('Error')
ax.set_facecolor('#C0C0C0')
plt.show()


# In[46]:


np.random.seed(8)
U = loadmat('icaTest.mat')['U']
A = loadmat('icaTest.mat')['A']
print(U.shape)
print(A.shape)
m,n = A.shape
X = A@U
W_init = np.random.rand(n,m)
W = ICA(X, W_init, learning_rate=0.01, MAXIMUM_ITERATIONS=1000000)
reconstructed_data = W@X
errors = la.norm(reconstructed_data-U,2)
print(f'Error: {errors}\n')


# In[47]:


plots(U, 1)


# In[48]:


plots(X, 1)


# In[49]:


v = np.array([reconstructed_data[0],reconstructed_data[2],-reconstructed_data[1]])
plots(v, 1)


# In[66]:


c = correlation_coefficient(U,v)
print(round(c,3))


# In[ ]:




