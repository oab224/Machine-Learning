import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class GradientDescent():
    def __init__(self):
        self.its = 100000
        self.eta = 1e-4
        self.eps = 1e-4

    def gradient_descent(self, X, y):
        theta = np.zeros((2, 1))
        cost_history = []
        m = len(y)
        #grad 1/m * XT @ predict
        for it in range(self.its):
            predict = X @ theta
            error = predict - y
            theta = theta - self.eta * X.T @ error / m
            cost_history.append(np.mean(error**2))
        #theta -= theta - eta * grad(np)
        return theta, cost_history
    
    def gradient(self, X, y, w):
        m = len(y)
        return  X.T.dot(X.dot(w) - y) / m
    
    def cost(self, w):
        return .5/ self.m * np.linalg.norm(self.X @ w - self.y, 2)**2
    
    def numerical_gradient(self, X, y, w):
        self.X = X
        self.y = y
        self.m = len(y)
        g = np.zeros_like(w)
        for i in range(len(g)):
            w_p = w.copy()
            w_n = w.copy()
            w_p[i] += self.eps
            w_n[i] -= self.eps
            g[i] = (self.cost(w_p) - self.cost(w_n)) / (2 * self.eps)
        return g
    
if __name__ == '__main__':
    print('algo')
    X = 2 * np.random.rand(100, 1)
    print('X: ', X)
    y = 4 + 3 * X + np.random.rand(100, 1)
    grad = GradientDescent()
    Xbar = np.c_[np.ones((100, 1)), X]
    print('Xbar: ', Xbar)
    theta, cost_history = grad.gradient_descent(Xbar, y)
    plt.title('Linear regression')
    x0 = np.linspace(min(X), max(X), 2)
    y0 = theta[0][0] + theta[1][0]*x0
    plt.plot(x0, y0)  
    print('theta: ', theta)

    #checking gradient
    w = np.random.rand(Xbar.shape[1], 1)
    print(np.linalg.norm(grad.gradient(Xbar, y, w) - grad.numerical_gradient(Xbar, y, w)) < 1e-6)

    plt.scatter(X, y, color='red', label = 'Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.plot(range(len(cost_history)), cost_history)
    plt.title('Cost History')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
