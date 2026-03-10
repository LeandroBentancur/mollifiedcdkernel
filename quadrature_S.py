# quadrature_S.py
#
# Originally written by Chiara Meroni, Jared Miller, and Mauricio Velasco
# as part of the auxiliary code for:
#   "Approximation of starshaped sets using polynomials"
#   https://github.com/ChiaraMeroni/polystar_bodies
#
# Used here with permission.

import numpy as np
import sympy as sp
import math as math
from functools import reduce
import operator

def Gegenbauer_poly(alpha : float, j: int, t):
    if j==0: return t**0
    if j==1: return 2*alpha*t
    if j>1:
        return (1/(j))*(2*t*(j+alpha-1)* Gegenbauer_poly(alpha, j-1,t) -(j+2*alpha-2)* Gegenbauer_poly(alpha,j-2,t))

def fast_evaluate_Gegenbauer_poly(alpha: float, j: int, t):
    assert j>=0, "Gegenbauer degree must be positive"
    minlen = max(j+1,2)
    results_array = np.zeros(minlen)
    results_array[0] = 1.0
    results_array[1] = 2*alpha*t
    #evaluate base cases if needed
    if j<=1:
        return results_array[j]
    else:
        #needs to use recursion   
        for k in range(2,j+1):
            n=k-1
            results_array[k] =  (2*(n+alpha)/(n+1))*t*results_array[k-1]-((n+2*alpha-1)/(n+1))*results_array[k-2]
    return results_array[j]

def test_gegenbauer_implementations_agreement():
    for degree in range(3,20):
        alpha = (degree-2)/2
        t=sp.symbols("t")
        gb = Gegenbauer_poly(alpha,degree,t)
        for tvalue in [0.0,0.1,0.5,0.7,0.9]:
            assert np.abs(fast_evaluate_Gegenbauer_poly(alpha,degree,tvalue)-gb.subs(t,tvalue))<1e-6

def Chebyshev_poly_1st_kind(j: int, t):
    if j==0: return t**0
    if j==1: return t
    if j>1:
        return 2*t*Chebyshev_poly_1st_kind(j-1, t) - Chebyshev_poly_1st_kind(j-2, t)

def Legendre_poly(j: int, t):
    if j==0: return t**0
    if j==1: return t
    if j>1:
        return (1/j)*((2*(j-1)+1)*t*Legendre_poly(j-1,t)-(j-1)*Legendre_poly(j-2,t))

#Computation of Gaussian quadrature weights via Golub-Welsh method
#We express the root finding problem as an eigenvalue estimation 
def beta(alpha,j):
    return np.sqrt(((j)*(j-1+2*alpha))/((j+alpha-1)*(j+alpha)))/2

def companion_matrix(alpha,k):
    T = np.zeros((k,k))
    for j in range(k):
        if j == 0:
            T[0,1]=beta(alpha,1) 
        elif j == k-1:
            T[k-1,k-2]=beta(alpha,k-1)
        else:
            T[j,j-1]=beta(alpha,j)
            T[j,j+1]=beta(alpha,j+1)
    return T

def weighted_Gaussian_Qrule_GW(alpha, degree: int):
    #Compute the roots of the one of degree d
    T = companion_matrix(alpha, degree)
    roots,eigenvectors = np.linalg.eig(T)
    weights = []
    #We compute the weights usign normalized eigenvectors
    moment_of_one=(math.pi*math.pow(2,1-2*alpha)*math.gamma(2*alpha))/(alpha*math.gamma(alpha)**2)
    for w in eigenvectors.T:
        weights.append((w[0]**2)*moment_of_one)
    return (roots,weights)

def weighted_Gaussian_roots_in_interval(numvars : int, degree_poly):
    assert numvars>=2,"roots for gaussian weights with alpha = (n-2)/2 or on circle"
    if numvars == 2:
        #we return the roots of the Chebyshev
        return np.array([np.cos(np.pi/(2*degree_poly)+k*(np.pi)/degree_poly) for k in range(degree_poly)])
    if numvars >=3:
        alpha = (numvars-2)/2
        roots, weights = weighted_Gaussian_Qrule_GW(alpha,degree_poly)
        return roots

#-----------------------------------------------------------------------
# Now spherical quadrature rule.
def circle_Qrule(degree : int):
    roots = []
    weights = []
    angle = 0
    for k in range(2*(degree+1)):
        newroot = np.zeros(2)
        newroot[0]=math.cos(angle)
        newroot[1]=math.sin(angle)
        roots.append(newroot)
        angle+=(2*math.pi)/(2*(degree+1))
        weights.append(math.pi/(degree+1))
    return roots, weights

def sphere_Qrule_inductive_step(current_nodes,current_weights, exactness_degree):
    #infer current number of variables from 0-th component of current_nodes
    numvars = len(current_nodes[0])
    alpha = (numvars-1)/2
    next_nodes = []
    next_weights = []
    gQ_roots, gQ_weights = weighted_Gaussian_Qrule_GW(alpha,exactness_degree)
    for sp_index, (root_sp,weight_sp) in enumerate(zip(current_nodes,current_weights)):
        for index_gQ,(root_GQ,weight_GQ) in enumerate(zip(gQ_roots,gQ_weights)):
            newroot = np.copy(root_sp)
            newroot = newroot * np.sqrt(1-root_GQ**2)
            newroot = np.pad(newroot,(1,0),"constant")
            newroot[0] = root_GQ
            next_nodes.append(newroot)
            next_weights.append(weight_sp*weight_GQ)
    return next_nodes,next_weights

#Sphere quadrature
def sphere_Quadrature(numvars, exactness_degree : int):
    exactness_degree = max(2,exactness_degree)
    roots, weights = circle_Qrule(exactness_degree) #A quadrature rule is specified by roots and weights
    assert numvars >= 2, "Implemented spherical quadrature requires positive dimension n-1"
    for k in range(2,numvars):
        roots, weights = sphere_Qrule_inductive_step(roots,weights,exactness_degree)#now on three-sphere
    return roots, weights

def integrate_with_quadrature(polynom, roots, weights, variables):
    res = 0.0
    for index, (root, weight) in enumerate(zip(roots,weights)):
        list_subs = list(zip(variables,root))
        term = polynom.subs(list_subs)*weight
        res+=term
    return res

def integrate_bbox_func_with_quadrature(box_func, roots, weights):
    #box_func is a function which given a point x returns f(x) 
    res = 0.0
    for index, (root, weight) in enumerate(zip(roots,weights)):
        term = box_func(root)*weight
        res+=term
    return res

def sphere_surface_area(numvars):
    assert numvars>=1, "dimension at least one"
    res = 0
    if numvars %2 == 0:
        res = 2*pow(math.pi, numvars/2)/math.factorial(numvars/2-1)
    if numvars %2 == 1:
        res = 2*pow(2*math.pi, (numvars-1)/2)/reduce(operator.mul, range(numvars-2,0,-2),1)
    return res