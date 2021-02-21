"""
    Example of using the fiducial NUTS sampler.
    Note --> for venv mishaps, try $hash -r
    Author: Alexander Murph
    Date: 2/14/21
"""
import os
from examples.fiducial_functions.multivariate_normal_fiducial_functions import *
import jax.numpy as np
import numpy as onp
from lib.FidHMC import FidHMC
from jax import random, ops
# import math
# import seaborn as sns
# import pandas as pd
import matplotlib.pyplot as plt
# from tensorflow_probability.substrates import jax as tfp
# os.environ['XLA_FLAGS'] = '--xla_dump_to=logfiles/'


def uncollapse_parameters(theta):
    parameter_dimension = int(0.5 * (-3 + np.sqrt(9 + 8 * len(theta))))
    a_matrix_dimension = int(0.5 * parameter_dimension * (parameter_dimension - 1))
    # Next, fill in A and Lambda in reverse of what is done in
    # collapseParameters.m
    a_matrix = np.zeros((parameter_dimension, parameter_dimension))
    lambda_matrix = np.diag(theta[a_matrix_dimension:(parameter_dimension + a_matrix_dimension)])
    mu_vector = theta[(parameter_dimension + a_matrix_dimension):]
    counter = 0
    for index_i in range(1, parameter_dimension):
        for j in range(index_i):
            a_matrix = ops.index_update(a_matrix, (index_i, j), theta[parameter_dimension + counter])
            a_matrix = ops.index_update(a_matrix, (j, index_i), -theta[parameter_dimension + counter])
            counter = counter + 1
    return a_matrix, lambda_matrix, mu_vector


def collapse_parameters(a_matrix, lambda_matrix, mu_vector):
    # Takes skew-symmetric matrix A, diagonal matrix Lambda, and a mu vector, and
    # collapses them into a single row vector of parameters theta.
    dimension = len(mu_vector)
    theta = np.concatenate((np.diag(lambda_matrix), mu_vector))

    # Grab the (d(d-1)/2) elements are from the lower triangle
    # of the matrix A.
    theta_temp = np.zeros(int(0.5 * (dimension * (dimension - 1))))
    counter = 0
    for index_i in range(1, dimension):
        for index_j in range(index_i):
            theta_temp = ops.index_update(theta_temp, ops.index[counter], a_matrix[index_i, index_j])
            counter = counter + 1
    theta = np.concatenate((theta_temp, theta))
    return theta


def run_example(seed):
    # Initial four draws from a MVN distribution:
    n = 4
    true_mu = np.asarray([1., 2., 3., 1.])
    true_Sigma = np.asarray([[4., 1., 0., 0.],
                             [1., 1., 0., 1.],
                             [0., 0., 9., 1.],
                             [0., 1., 1., 4.]]).astype(float)
    Lambda_0, u = np.linalg.eig(true_Sigma)
    Lambda_0 = np.diag(Lambda_0 ** 0.5)
    dim = len(true_mu)
    int_string = "%0" + str(dim) + "d"
    my_count = 0
    originalsign = np.sign(np.linalg.det(u)).astype(int)
    for i in range((2 ** dim - 1)):
        bits_string = int_string % int(bin(i)[2:])
        temp_signs = np.asarray(2 * onp.asarray(list(bits_string)).astype(int) - 1)
        if 1 == originalsign * np.prod(temp_signs):
            temp_u = np.matmul(u, np.diag(temp_signs.astype(float)))
            temp_Am = np.linalg.solve((np.identity(dim) + temp_u), (np.identity(dim) - temp_u))
            temp_Av = np.tril(temp_Am)
            magnitude_less_than_1 = np.asarray([[-1. < x < 1. for x in y] for y in temp_Av]).reshape(1, dim ** 2)
            if np.all(magnitude_less_than_1):
                u = temp_u
                my_count = my_count + 1
    A_0 = np.linalg.solve(np.identity(4) + u, np.identity(4) - u)
    true_theta = collapse_parameters(A_0, Lambda_0, true_mu).astype(float)

    # Pick a proposal A matrix whose values are between -1 and 1.
    tempData = random.multivariate_normal(random.PRNGKey(seed), true_mu, true_Sigma, shape=[n])
    tempCovariance = (float(n)) ** (-1.) * np.matmul(tempData.transpose(), tempData)
    Lambda_0, u = np.linalg.eig(tempCovariance)
    Lambda_0 = np.diag(Lambda_0 ** 0.5)
    dim = len(true_mu)
    int_string = "%0" + str(dim) + "d"
    my_count = 0
    originalsign = np.sign(np.linalg.det(u)).astype(int)
    for i in range((2 ** dim - 1)):
        bits_string = int_string % int(bin(i)[2:])
        temp_signs = np.asarray(2 * onp.asarray(list(bits_string)).astype(int) - 1)
        if 1 == originalsign * np.prod(temp_signs):
            temp_u = np.matmul(u, np.diag(temp_signs.astype(float)))
            temp_Am = np.linalg.solve((np.identity(dim) + temp_u), (np.identity(dim) - temp_u))
            temp_Av = np.tril(temp_Am)
            magnitude_less_than_1 = np.asarray([[-1. < x < 1. for x in y] for y in temp_Av]).reshape(1, dim ** 2)
            if np.all(magnitude_less_than_1):
                u = temp_u
                my_count = my_count + 1
    A_0 = np.linalg.solve(np.identity(4) + u, np.identity(4) - u)

    # Establish true parameters, data, and initial theta value
    data_0 = random.multivariate_normal(random.PRNGKey(seed), true_mu, true_Sigma, shape=[n])
    theta_0 = collapse_parameters(A_0, Lambda_0, true_mu)
    lower_bounds = [-1., -1., -1., -1., -1., -1., None, None, None, None, None, None, None, None]
    upper_bounds = [1., 1., 1., 1., 1., 1., None, None, None, None, None, None, None, None]
    my_path = os.path.dirname(os.path.abspath(__file__))
    # np.save(my_path + "/data/MVN_trueA.npy", a)
    # np.save(my_path + "/data/MVN_trueLambda.npy", l)
    # np.save(my_path + "/data/MVN_trueMu.npy", m)

    # Create the object and perform NUTS:
    fhmc = FidHMC(log_likelihood, dga_func, eval_func, len(theta_0), data_0, lower_bounds, upper_bounds)
    states, log_probs = fhmc.run_NUTS(num_iters=2, burn_in=1, initial_value=theta_0, random_key=random.PRNGKey(seed))

    # Save the data if using simulation study:
    # np.save(my_path + "/data/simulations/MVN_States_" + os.getenv('SLURM_ARRAY_TASK_ID') + ".npy", states)

    # Save the data:
    # np.save(my_path + "/data/MVN_States.npy", states)
    # np.save(my_path + "/data/MVN_LogProbs.npy", log_probs)


def create_plots():
    # Get Parameter Names
    my_path = os.path.dirname(os.path.abspath(__file__))
    true_a = np.load(my_path + "/data/MVN_trueA.npy")
    true_lambda = np.load(my_path + "/data/MVN_trueLambda.npy")
    true_mu = np.load(my_path + "/data/MVN_trueMu.npy")

    # Load the data
    my_path = os.path.dirname(os.path.abspath(__file__))
    states = np.load(my_path + "/data/SimpleNormal_States.npy")
    log_probs = np.load(my_path + "/data/SimpleNormal_LogProbs.npy")

    # Graph the log probability
    plt.figure()
    plt.plot(log_probs)
    plt.ylabel('Target Log Prob')
    plt.xlabel('Iterations of NUTS')
    plt.savefig(my_path+'/plots/MVN_mcmc_log_probability.png')

# int(os.getenv('SLURM_ARRAY_TASK_ID'))
run_example(13)
create_plots()
