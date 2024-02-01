import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

def one_hot_vector(y, c):
    return sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape=(c, len(y))).toarray()  

def softmax(z):
    # e^z/sigma(e^z)
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / e_z.sum(axis = 0)

def cost(theta, X, y):
    a = softmax(theta.T @ X)
    return -np.sum(y * np.log(a))  # Take the mean along the samples


def grad(theta, X, y):
    return X @ (softmax(theta.T @ X) - y).T 

def SGD(theta, X, y, eta, tol=1e-4, max_count=10000):
    N = X.shape[1]
    d = X.shape[0]
    c = theta.shape[1]
    count = 0 
    theta_list = [theta]
    check_theta_after = 20 #batch size
    while (count < max_count):
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi =  X[:, i].reshape(d, 1) #shape (d x 1)
            yi =  y[:, i].reshape(c, 1)  #shape (c x 1)
            theta_new = theta_list[-1] - eta*grad(theta_list[-1], xi, yi)
            theta_list.append(theta_new)

            count += 1
            if count % check_theta_after == 0:
                if np.linalg.norm(theta_new - theta_list[-check_theta_after]) < tol:
                    return theta_list
            
    return theta_list

def numerical_grad(theta, X, y, cost):
    eps = 1e-6
    g = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[i, j] += eps
            theta_m[i, j] -= eps
            g[i, j] = (cost(theta_p, X, y) - cost(theta_m, X, y)) / (2 * eps)

    return g

def pred(theta, X):
    return np.argmax(softmax(theta.T @ X), axis = 0)

def draw(X, label):
#     K = np.amax(label) + 1
    X0 = X[:, label == 0]
    X1 = X[:, label == 1]
    X2 = X[:, label == 2]
    
    plt.plot(X0[0, :], X0[1, :], 'b^', markersize=4, alpha=0.8, markeredgewidth=0.5, markeredgecolor='black')
    plt.plot(X1[0, :], X1[1, :], 'go', markersize=4, alpha=0.8, markeredgewidth=0.5, markeredgecolor='black')
    plt.plot(X2[0, :], X2[1, :], 'rs', markersize=4, alpha=0.8, markeredgewidth=0.5, markeredgecolor='black')
#     plt.axis('equal')
    plt.axis('off')
    plt.plot()
    plt.show()
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
print(X0.shape)
X = np.concatenate((X0, X1, X2), axis = 0).T # each column is a datapoint
X = np.concatenate((np.ones((1, 3*N)), X), axis = 0)
C = 3
y = np.asarray([0]*N + [1]*N + [2]*N)

np.random.seed(1)
theta = np.random.rand(X.shape[0], C)
y_h = one_hot_vector(y, C)
theta = SGD(theta, X, y_h, .05)[-1]

print(theta)

x = np.arange(-2, 11, .025)
_y = np.arange(-3, 10, .025)
xx, yy = np.meshgrid(x, _y)

x_train = xx.ravel().reshape(1, xx.size)
y_train = yy.ravel().reshape(1, yy.size)
X_train = np.concatenate((np.ones((1, x_train.size)), x_train, y_train), axis=0)

Z = pred(theta, X_train)

Z = Z.reshape(xx.shape)
CS = plt.contourf(xx, yy, Z, 300, cmap = 'jet', alpha= 0.1)
plt.xlim(-2, 11)
plt.ylim(-3, 10)
plt.xticks(())
plt.yticks(())
draw(X[1:, :], y)


plt.show()