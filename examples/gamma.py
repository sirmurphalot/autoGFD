"""
    Example of using the fiducial NUTS sampler.
    Note --> for venv mishaps, try $hash -r
    use: python -m examples.simple_normal
    Author: Alexander Murph
    Date: 2/14/21
"""
import os
from examples.fiducial_functions.gamma_fiducial_functions import *
import jax.numpy as np
from lib.FidHMC import FidHMC
from jax import random
import seaborn as sns
import pandas as pd
import time
import scipy
import numpy as onp
import matplotlib.pyplot as plt

global true_theta
global n
global number_of_iters
global number_of_burnin
global number_of_chains
true_theta = [2., 3.]
n = 50
number_of_iters = 150000
number_of_burnin = 50000
number_of_chains = 4


def run_example():
    # Establish true parameters, data, and initial theta value:
    data_0 = scipy.stats.gamma.rvs(true_theta[0], scale=(1/true_theta[1]), size=n)
    theta_0 = np.asarray([1., 1.])
    lower_bounds = [0., 0.]
    upper_bounds = [None, None]

    # Create the object and perform NUTS:
    t0 = time.time()
    fhmc = FidHMC(log_likelihood, None, None, 2, data_0, lower_bounds, upper_bounds, number_of_cores=1,
                  user_l2_jac_det_term=l2_jac_func)
    # With bounds:
    # states, log_accept = fhmc.run_NUTS(num_iters=number_of_iters, burn_in=number_of_burnin, initial_value=theta_0, step_size=15e-2,
    #                                   num_chains=number_of_chains)
    states, log_accept = fhmc.run_RWM(num_iters=number_of_iters, burn_in=number_of_burnin,
                                      initial_value=theta_0, proposal_scale=1e-2)
    # states, log_accept = fhmc.run_HMC(num_iters=150, burn_in=50, initial_value=theta_0, step_size=15e-2)
    t1 = time.time()

    # states, [new_step_size, log_accept] = fhmc.run_HMC(num_iters=1500, burn_in=1500, step_size=1e-2)
    print("---------------------------------")
    print("MCMC draw complete.")
    print("Acceptance Ratio: ", np.exp(np.log(np.mean(np.exp(np.minimum(log_accept, 0.))))))
    # print("Acceptance Ratio: ", np.mean(log_accept))
    # print("Final Step Size: ", new_step_size)
    print("Execultion time: ", t1 - t0)
    print("---------------------------------")

    # Save the data:
    my_path = os.path.dirname(os.path.abspath(__file__))
    np.save(my_path + "/data/Gamma_States.npy", states)
    np.save(my_path + "/data/Gamma_AcceptanceRatio.npy",
            np.exp(np.log(np.mean(np.exp(np.minimum(log_accept, 0.))))))
    np.save(my_path + "/data/Gamma_ExecutionTime.npy", np.array(t1 - t0, float))
    np.save(my_path + "/data/Gamma_RawData.npy", np.array(data_0))

def graph_results():
    # Get Parameter Names
    col_names = []
    for d in range(len(true_theta)):
        col_names.append("theta_" + str(d))

    # Load the data
    my_path = os.path.dirname(os.path.abspath(__file__))
    states = np.load(my_path + "/data/Gamma_States.npy")
    accept_ratio = np.load(my_path + "/data/Gamma_AcceptanceRatio.npy")
    execution_time = np.load(my_path + "/data/Gamma_ExecutionTime.npy")
    data_0 = np.load(my_path + "/data/Gamma_RawData.npy")

    # Get point estimates
    mean = np.mean(data_0)
    var = np.var(data_0)
    point_estimates = [mean**2/var, mean/var]

    # Graph the parameter draws
    # states = states[(np.abs(stats.zscore(states)) < 3.).all(axis=1)]
    temp_sample_df = pd.DataFrame(states)
    sample_df = temp_sample_df.melt()

    g = sns.displot(data=sample_df, x="value", row="variable", kind="kde", fill=1, color="blue",
                    height=2.5, aspect=3, facet_kws=dict(margin_titles=True), )

    count = 0
    for ax in g.axes.flat:
        ax.axvline(true_theta[count], color="red")
        ax.axvline(point_estimates[count], color="purple")
        count += 1
    g.fig.suptitle("Acceptance Ratio: " + str(accept_ratio) + ", Execution Time: " + str(execution_time))
    g.savefig(my_path + '/plots/Gamma_mcmc_samples.png')

    # Simulate values from the true distribution and compare them to the values sampled by MCMC chain.


#run_example()
graph_results()
