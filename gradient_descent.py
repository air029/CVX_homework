# title: Gradient descent
# author: Jiahong Ma
# data: 2020/11/24

import math
import numpy as np
from matplotlib import pyplot as plt


def f(x):
    x1 = x[0]
    x2 = x[1]
    return math.exp(x1 + 3 * x2 - 0.1) + math.exp(x1 - 3 * x2 - 0.1) + math.exp(-x1 - 0.1)

def grad(x):
    x1 = x[0]
    x2 = x[1]
    gradx1 = math.exp(x1 + 3 * x2 - 0.1) * 1 + math.exp(x1 - 3 * x2 - 0.1) * 1 + math.exp(-x1 - 0.1) * -1
    gradx2 = math.exp(x1 + 3 * x2 - 0.1) * 3 + math.exp(x1 - 3 * x2 - 0.1) * -3
    return np.asarray([gradx1, gradx2])

def norm(grad):
    gradx1 = grad[0]
    gradx2 = grad[1]
    return (gradx1 ** 2 + gradx2 ** 2) ** 0.5

def gradient_descent(alpha, beta, epsilon, x0):
    assert (alpha > 0 and alpha < 0.5)
    assert (beta > 0 and beta < 1)

    x = [x0]
    d = [-1 * grad(x0)]
    k = 0

    while (norm(grad(x[k])) > epsilon):
        t = 1
        rawf = f(x[k] + t * d[k])
        newf = f(x[k]) + alpha * t * np.dot(grad(x[k]), d[k])
        while (rawf > newf):
            t = beta * t
            rawf = f(x[k] + t * d[k])
            newf = f(x[k]) + alpha * t * np.dot(grad(x[k]), d[k])

        newx = x[k] + t * d[k]
        x.append(newx)
        d.append(-1 * grad(newx))
        k = k + 1

    return x

def distance(result):
    dist = []
    for x in result:
        dist.append(f(x) - TRUTH)
    return dist

def alpha_exp(k):
    alpha = k
    betalist = [0.1,  0.3, 0.5, 0.7, 0.9]
    dist = []
    for x in betalist:
        tempresult = (gradient_descent(alpha = alpha, beta = x, epsilon = 0.35, x0 = np.asarray([0.45,0.45])))
        dist.append(distance(tempresult))
    return dist,betalist

def beta_exp(k):
    beta = k
    alphalist = [0.1, 0.2, 0.3, 0.4]
    dist = []
    for x in alphalist:
        tempresult = (gradient_descent(alpha = x, beta = beta, epsilon = 0.1, x0 = np.asarray([0.12,0.12])))
        dist.append(distance(tempresult))
    return dist,alphalist


#--------------------------------------------truth------------------------------------------
t = gradient_descent(alpha = 0.3, beta = 0.3, epsilon = 0.000000001, x0 = np.asarray([1,1]))
TRUTH = f(t[len(t)-1])


for x in t:
    print('delta:')
    print(f(x)-TRUTH)

