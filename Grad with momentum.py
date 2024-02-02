import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from sklearn import linear_model

np.random.seed(2)
X = np.random.rand(1000, 1)
y = 4 + 3*X + .2*np.random.rand(1000, 1)
Xbar = np.c_[np.ones((1000, 1)), X]
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y)
exact_w = regr.coef_
N = len(y)
lamda = 0.9
eta = 0.5
its = 1000
#print(Xbar)

#calculate gradient
def gradient(theta):
    return 1/N*Xbar.T@(Xbar.dot(theta) - y)


def gradient_with_momentum():
    theta_new = np.array([[2], [1]])  
    v_old = np.zeros((2, 1))
    theta = [theta_new]
    it = 0
    for i in range(100):
        v_new = lamda*v_old + eta*gradient(theta_new)
        theta_new = theta_new - v_new
        v_old = v_new
        theta.append(theta_new)
        #checking covergence threshold
        if np.linalg.norm(gradient(theta_new)) / len(theta_new) < 1e-3:
            break
    return (theta, i)




#draw

#parameters
a1 = np.linalg.norm(X)**2/N
b1 = np.linalg.norm(y)**2/N
c1 = 2*np.sum(X)/N
d1 = -2*np.sum(y)/N
e1 = -2*X.T@y/N

#grid 
xg = np.arange(1.5, 7.0, 0.025)
yg = np.arange(0.5, 7.0, 0.025)
w0, w1 = np.meshgrid(xg, yg)

#equation
Z = a1*w1**2 + w0**2 + b1 + c1*w1*w0 + d1*w0 + e1*w1

#define fig,ax 
fig, ax = plt.subplots(figsize=(4, 4))

theta, its = gradient_with_momentum()

#contour draw
def update(frame):
    if frame == 0:
        plt.cla() #clear screen
        CT = plt.contour(w0, w1, Z, 100)
        locations = [(4.15, 3.5), (4.25, 3.75), (4.5, 4)]
        animlist = plt.clabel(CT, inline = .1, fontsize = 10, manual = locations)
        plt.plot(exact_w[0][0], exact_w[0][1], 'go')
    else:
        animlist = plt.plot([theta[frame-1][0], theta[frame][0]], [theta[frame-1][1], theta[frame][1]], 'r-')
    animlist = plt.plot(theta[frame][0], theta[frame][1], 'ro', markersize = 4)
    xlabel = '$\eta =$ ' + str(eta) + '; iter = %d/%d' %(frame, its)
    xlabel += '; ||grad||_2 = %.3f' % np.linalg.norm(gradient(theta[frame]))
    ax.set_xlabel(xlabel)
    return animlist, ax
anim = FuncAnimation(fig, update, frames = np.arange(0, its), interval = 200)
fn = 'LR_momentum_contours.gif'
anim.save(fn, dpi=100, writer='imagemagick')
plt.show()
