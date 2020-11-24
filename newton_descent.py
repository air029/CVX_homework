# title: Newton descent
# author: Jiahong Ma
# data: 2020/11/24

from __future__ import division
import numpy as np
import random
import math
from goto import with_goto



#-----------------------------------------func--------------------------------------------
def init_engine_x(n):
    x = []
    for i in range(n):
        #x.append(random.randint(1,4))
        x.append(random.uniform(0, 3))
    return np.array(x)

def rank_test(matrix, p):
    rank = np.linalg.matrix_rank(matrix)
    if(rank == p):
        return True
    else:
        return False

def init_engine_A( p, n):
    while(True):
        temp = np.random.randint(1, 4, (p, n))
        if(rank_test(temp, p) == True):
            return np.array(temp)

def init_engine_xhat(n):
    xhat = []
    for i in range(n):
        xhat.append(np.random.uniform(0,2))
    return np.array(xhat)

def fx(x):
    sum = 0
    for num in x:
        sum += num * math.log(num)
        #sum += num * np.log(num)
    return sum


def gradf(x):
    temp = []
    for num in x:
        temp.append(num * math.log(num))
        #temp.append(num * np.log(num))
    return (np.array(x))


def hessianf(x):
    temp = []
    for num in x:
        temp.append(1/num)
    return np.diag(temp)


def b(A, xhat):
    return A.dot(xhat)

def r(x, v, A, b):
    line1 = np.array(gradf(x) + np.dot(A.T,v))
    #print(line1.shape)
    line2 = A.dot(x) - b
    #print(line2.shape)
    r = np.concatenate((line1,line2), axis=0)
    return r


def equal(a,b):
    if(math.fabs(a-b) <= 1e-8):
        return True
    else:
        return False


#-------------------------------------------Newton algorithm------------------------------------------------
#algo 1-------------------------------------------------------------------------------------
@with_goto
def algorithm1(alpha, beta, epsilon, x0, A):
    #step1
    x = [x0]
    k = 0

    #step2
    label .flagA1

    line1 = np.concatenate((hessianf(x[k]), A.T), axis=1)
    line2 = np.concatenate((A, np.zeros((30, 30))), axis=1)
    MATRIX_A = np.concatenate((line1, line2), axis=0)

    MATRIX_B = np.concatenate((-1 * gradf(x[k]), np.zeros((30,))), axis=0)

    MATRIX_X = np.linalg.solve(MATRIX_A, MATRIX_B)

    dx = MATRIX_X[:100]

    lam_2 = dx.dot(hessianf(x[k])).dot(dx)

    #step3
    if((lam_2 / 2) <= epsilon):
        return x

    #step4
    tk = 1
    while(fx(x[k] + tk * dx)) > fx(x[k]) - alpha * tk * lam_2:
        tk = beta * tk

    #step5
    newx = x[k] + tk * dx
    x.append(newx)


    print(fx(newx))

    if(fx(x[k]) == fx(newx)):
        return x
    k += 1

    goto .flagA1

#algo 2-------------------------------------------------------------------------------------
@with_goto
def algorithm2(alpha, beta, epsilon, x0, v0, A, b):
    #step1
    x = [x0]
    v = [v0]
    k = 0

    #step2
    label .flagB1
    testA = (A.dot(x[k]) - b == 0).all()
    testB = (np.linalg.norm(r(x[k],v[k],A,b)) <= epsilon)
    if(testA and testB):
          return x

    #step3
    line1 = np.concatenate((hessianf(x[k]), A.T), axis=1)
    line2 = np.concatenate((A, np.zeros((30, 30))), axis=1)
    MATRIX_A = np.concatenate((line1, line2), axis=0)

    MATRIX_B = -1 * np.concatenate((gradf(x[k]) + A.T.dot(v[k]), A.dot(x[k])-b), axis=0)

    MATRIX_X = np.linalg.solve(MATRIX_A, MATRIX_B)


    dx = np.array(MATRIX_X[0:100])
    dv = np.array(MATRIX_X[100:130])


    #step4
    tk = 1
    while(np.linalg.norm(r(x[k] + tk * dx, v[k] + tk * dv, A, b)) >= (1-alpha*tk)*np.linalg.norm(r(x[k], v[k], A, b))):
        tk = beta * tk

    #step5
    newx = x[k] + tk * dx
    x.append(newx)
    newv = v[k] + tk * dv
    v.append(newv)

    print(fx(newx))

    if(equal(fx(x[k]),fx(newx))):
        return x
    k += 1

    goto .flagB1



#algo 3-------------------------------------------------------------------------------------
def g_(v, b, A):
    bv = b.T.dot(v)
    minus_Av = -1 * A.T.dot(v)
    fdual = 0
    for x in minus_Av:
        fdual += math.exp(x-1)
    return bv + fdual


def graddual(vec):
    temp = []
    for x in vec:
        temp.append(math.exp(x-1))
    return np.array(temp)

def hessiandual(vec):
    temp = []
    for x in vec:
        temp.append(math.exp(x-1))
    return np.diag(temp)

def gradg(v, A, b):
    return -b + A.dot(graddual(-1 * A.T.dot(v)))

def hessiang(v, A):
    return A.dot(hessiandual(-1 * A.T.dot(v)).dot(A.T))

def gradg_(v, A, b):
    return -1 * gradg(v, A, b)

def hessiang_(v, A):
    return hessiang(v, A)


@with_goto
def algorithm3(alpha, beta, epsilon, v0, A, b):
    #step1
    v = [v0]
    k = 0

    #step2
    label .flagC1

    dv = -1 * np.dot(np.linalg.inv(hessiang_(v[k], A)), gradg_(v[k], A, b))
    lam_2 = (dv).dot(hessiang_(v[k], A)).dot(dv)

    #step3
    if((lam_2 / 2) <= epsilon):
        return v

    #step4
    tk = 1
    while(g_(v[k] + tk * dv, b, A) > g_(v[k], b, A) - alpha * tk * lam_2):
        tk = beta * tk

    #step5
    newv = v[k] + tk * dv
    v.append(newv)

    # print('g_()--------------------------------------------------------------')
    print(g_(newv, b, A))



    k += 1

    goto .flagC1




#-----------------------------------------main----------------------------------------------------


#------------------------------------data-----------------------------
A = init_engine_A(30, 100)
xhat = init_engine_xhat(100)
b = A.dot(xhat)
v = np.zeros((30,))
#v = init_engine_x(30)



print('-------------------------------algo1-------------------------')
result1 = algorithm1(alpha=0.25, beta=0.8, epsilon=0.001, x0=xhat, A=A)



print('-------------------------------algo2-------------------------')
algorithm2(alpha=0.25, beta=0.8, epsilon=0.001, x0=xhat, v0=v, A=A, b=b)



print('-------------------------------algo3-------------------------')
algorithm3(alpha=0.25, beta=0.5, epsilon=0.0001, v0=v, A=A, b=b)
