# quadrature_S.py
#
# Originally written by Chiara Meroni, Jared Miller, and Mauricio Velasco
# as part of the auxiliary code for:
#   "Approximation of starshaped sets using polynomials"
#   https://github.com/ChiaraMeroni/polystar_bodies
#
# Used here with permission.

import numpy as np
import math as math

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