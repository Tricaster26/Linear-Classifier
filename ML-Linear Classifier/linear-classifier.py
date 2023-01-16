import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
f = open('two-L.pkl', 'rb')
pos, neg = pickle.load(f)
x_axis = []
f.close()
#make into numpy arrays
n_pos = np.array(pos, dtype=float)
n_neg = np.array(neg, dtype=float)



data_matrix_X = np.concatenate((n_pos,n_neg),axis=0,dtype=float)
true_output_y = np.append([1]*550,[0]*550)

m,n = data_matrix_X.shape
#there are only 2 features, so dimensions of w is 1 x 2
w = np.zeros((1,n))
b = 0
step_size = 10
#sigmoid function
def predict(x):
    return 1.0/(1 + np.exp(-x))
#100 runs
for i in range (1000):
    for i in range(len(data_matrix_X)):
        pred_y = predict(np.dot(w, np.array([data_matrix_X[i]]).T) + b)[0][0]
        grad_w = ((1/m)*np.dot( (pred_y - true_output_y[i]),np.array([data_matrix_X[i]])))
        grad_b = (1/m)*(pred_y - true_output_y[i])
        w = w - step_size*grad_w
        b = b - step_size*grad_b

#math in latex doc
m = -w[0][0]/w[0][1]
c =-b/w[0][1]
fig = plt.figure(figsize=(8,6))
plt.scatter(n_neg[:,0],n_neg[:,1], label = '-1')
plt.scatter(n_pos[:,0],n_pos[:,1], label = '+1')
plt.legend(loc="upper right")
x = np.linspace(-0.25,2,100)
plt.plot(x , m*x + c, color = 'green')

def mean(x):
    # make it a 2d vector
    avg = np.array([[0,0]],dtype=float)
    for i in range(len(x)):
        avg[0][0] +=  x[i][0]
        avg[0][1] +=  x[i][1]
    avg[0][0] = avg[0][0]/len(x)
    avg[0][1] = avg[0][1]/len(x)
    return avg
mean(n_pos)
def var(x):
    var = np.array([[0,0],[0,0]], dtype=float)
    mean_x = mean(x)
    for i in range(len(x)):
        var[0][0] += (x[i][0] - mean_x[0][0])*(x[i][0] - mean_x[0][0])
        var[0][1] += (x[i][0] - mean_x[0][0])*(x[i][1]-mean_x[0][1])
        var[1][0] += (x[i][1]-mean_x[0][1])*(x[i][0] - mean_x[0][0])
        var[1][1] += (x[i][1]-mean_x[0][1])*(x[i][1]-mean_x[0][1])
    var[0][0] = var[0][0]/len(x)
    var[0][1] = var[0][1]/len(x)
    var[1][0] = var[0][1] /len(x)
    var[1][1] = var[1][1]/len(x)
    return var

u_1 = mean(n_pos)
u_2 = mean(n_neg)
var_1 = var(n_pos)
var_2 = var(n_neg)

print("u1: ",u_1 ,"u2: ", u_2,"v1: ", var_1,"v2: ",var_2)

u_ultimate = (u_1 + u_2)/2
u_ultimate

u_perpendicular = (u_2 - u_1)

#gradient of classifier is negative inverse of gradient of u_perpendicular vector

fig = plt.figure(figsize=(8,6))
plt.scatter(n_neg[:,0],n_neg[:,1], label = '-1')
plt.scatter(n_pos[:,0],n_pos[:,1], label = '+1')
plt.plot(u_2[0][0],u_2[0][1],'.',color = 'red', label = 'u_2')
plt.plot(u_1[0][0],u_1[0][1],'.',color = 'black', label = 'u_1')
x = np.linspace(-0.25,2,100)
#lines dont look perpendicular because of unequal axes.
plt.plot(x , 0.353*x -0.353, color = 'green')
# plt.plot(x , -2.8329*x , color = 'green')
plt.legend(loc="upper right")
