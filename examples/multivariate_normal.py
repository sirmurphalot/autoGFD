"""
    Example of using the fiducial NUTS sampler.
    Note --> for venv mishaps, try $hash -r
    Author: Alexander Murph
    Date: 2/14/21
"""
import os
from examples.fiducial_functions.simple_normal_fiducial_functions import *
import jax.numpy as np
from lib.FidHMC import FidHMC
from jax import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

n = 4
mu = np.asarray([-1., 1., -1., 1.])
tempSigma = np.diag(np.asarray(range(4)).astype(float)+1.)*10.
tempData = random.multivariate_normal(random.PRNGKey(13), mu, tempSigma, shape=[n])
tempCovariance = (float(n))**(-1.)*np.matmul(tempData.transpose(), tempData)
Lambda_0, u =np.linalg.eig(tempCovariance)
u_initial=u
Lambda_0= np.diag(Lambda_0 ** 0.5)

dim = len(mu)
int_string = "%0" + str(dim) + "d"
my_count=0
originalsign=np.sign(np.linalg.det(u))
for i in range((2**dim-1)):
    bits_string= int_string % int(bin(i)[2:])
    temp_signs = np.asarray(list(bits_string)).astype(int) - 1
    if 1==originalsign*np.prod(temp_signs):
        temp_u=u*np.diag(temp_signs)
        temp_Am=np.linalg.solve((np.identity(dim)+temp_u), (np.identity(dim)-temp_u))
        temp_Av=np.tril(temp_Am)
        if (1==np.prod(temp_Av>-1 & temp_Av<1)):
            Av=temp_Av
            u=temp_u
            my_i=i
            my_count=my_count+1

A_0 = np.linalg.solve(np.identity(4)+u, np.identity(4)-u)
q0=collapseParameters(A_0,Lambda_0)
Data = random.multivariate_normal(random.PRNGKey(13), mu,
    np.matmul(u, np.matmul(Lambda_0, u.transpose())), shape=[n])

# Establish true parameters, data, and initial theta value
true_theta = [-0.5, 3.2, 1.0, 1., 1., 1.]
data_0 = random.multivariate_normal(random.PRNGKey(13), np.asarray([-0.5, 3.2, 1.0]), np.identity(3), shape=[2])
theta_0 = np.asarray([1., 1., 1., 1., 1., 1.])

# Create the object and perform NUTS:
fhmc = FidHMC(log_likelihood, dga_func, eval_func, 6, data_0)
states, log_probs = fhmc.run_NUTS(num_iters=15000, burn_in=5000, initial_value=theta_0)