import pygame
import numpy as np
import random
from sys import exit
import matplotlib.pyplot as plt
from sklearn import linear_model
pygame.init()
surface = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()

pygame.display.flip()
pygame.display.set_caption('K-means algorithm')

_font = pygame.font.Font(None, 40)
_run_rect = _font.render('F5 to run', False, 'black')
data_points = []



surface.fill((133, 158, 164))
clock.tick(60)

label = []

    
def algo():
    ndata_points = np.array(data_points)
    X = np.array([ndata_points[:, 0]]).T
    y = np.array([ndata_points[:, 1]]).T
   
    plt.title("Linear Regression")
    plt.plot(X, y, 'ro')
    plt.axis([0, 770, 0, 530])
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('top')
    plt.xlabel('X')             
    plt.ylabel('y')
    #
    #building line
    one = np.ones((X.shape[0], 1))
    xBar = np.concatenate((one, X), axis = 1)
    print(xBar)
    #XT.X.W = XT.Y <=> W = (XT.X)pseudoinv.XT.Y
    W = np.dot(np.dot(np.linalg.pinv(np.dot(xBar.T, xBar)), xBar.T), y)
    x0 = np.linspace(0, 770, 2)
    y0 = W[0][0] + W[1][0]*x0
    plt.plot(x0, y0)     
    print(W)
    regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    regr.fit(xBar, y)
    print("\n\nalgo")
    print(regr.coef_)
    print("\n\nimplements")
    print(W)
    plt.show()

Rdom = False

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    #check coordinate of data points
        if event.type == pygame.MOUSEBUTTONDOWN:
            if (event.pos[0] >= 30 and event.pos[0] <= 770) and (event.pos[1] >= 30 and event.pos[1] <= 530):
                data_points.append(event.pos)

    #plus minus clusters
        if event.type == pygame.KEYDOWN:
            #run algorithm
            if event.key == pygame.K_F5:
                #if already have random clusters
                print(data_points)
                algo()
            if event.key == pygame.K_F6:
                #if already have random clusters
                data_points = []
          
    pygame.draw.rect(surface, 'white', pygame.Rect(30, 30, 740, 500))            

    for i in range(len(data_points)):
        if label == []:
            pygame.draw.circle(surface, 'black', data_points[i], 5)
            

    

            
    #draw surface
    pygame.draw.rect(surface, (133, 158, 164), pygame.Rect(0, 530, 800, 200))
    surface.blit(_run_rect, _run_rect.get_rect(midtop = (400, 0)))
   
    pygame.display.flip()

 