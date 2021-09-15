"""
    Example of using the fiducial NUTS sampler.
    Note --> for venv mishaps, try $hash -r
    use: python -m examples.simple_normal
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
import time
import scipy
import numpy as onp
import matplotlib.pyplot as plt

global true_theta
global n
global number_of_iters
global number_of_burnin
global number_of_chains
true_theta = [-0.5, 3.2, 1.0, 1.9, 1.1, 2.5]
n = 50
number_of_iters = 15000
number_of_burnin = 5000
number_of_chains = 4


def run_example():
    # Establish true parameters, data, and initial theta value:
    data_0 = random.multivariate_normal(random.PRNGKey(13), np.asarray(true_theta[0:3]),
                                        np.diag(np.asarray(true_theta[3:])), shape=[n])
    theta_0 = np.asarray([1., 1., 1., 1., 1., 1.])
    lower_bounds = [None, None, None, 0., 0., 0.]
    upper_bounds = [None, None, None, None, None, None]

    # Create the object and perform NUTS:
    t0 = time.time()
    fhmc = FidHMC(log_likelihood, dga_func, eval_func, 6, data_0, lower_bounds, upper_bounds, number_of_cores=40)
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
    np.save(my_path + "/data/SimpleNormal_States.npy", states)
    np.save(my_path + "/data/SimpleNormal_AcceptanceRatio.npy",
            np.exp(np.log(np.mean(np.exp(np.minimum(log_accept, 0.))))))
    np.save(my_path + "/data/SimpleNormal_ExecutionTime.npy", np.array(t1 - t0, float))
    np.save(my_path + "/data/SimpleNormal_RawData.npy", data_0)


def graph_results():
    # Grab the original data
    my_path = os.path.dirname(os.path.abspath(__file__))
    data_0 = np.load(my_path + "/data/SimpleNormal_RawData.npy")
    ybar1 = np.mean(data_0[:, 0])
    ybar2 = np.mean(data_0[:, 1])
    ybar3 = np.mean(data_0[:, 2])

    sd1 = np.std(data_0[:, 0], ddof=1)
    sd2 = np.std(data_0[:, 1], ddof=1)
    sd3 = np.std(data_0[:, 2], ddof=1)

    point_estimators = [ybar1, ybar2, ybar3, sd1**2, sd2**2, sd3**2]

    # Get Parameter Names
    col_names = []
    for d in range(len(true_theta)):
        col_names.append("theta_" + str(d))

    # Load the data
    states = np.load(my_path + "/data/SimpleNormal_States.npy")
    accept_ratio = np.load(my_path + "/data/SimpleNormal_AcceptanceRatio.npy")
    execution_time = np.load(my_path + "/data/SimpleNormal_ExecutionTime.npy")

    # Graph the parameter draws
    # states = states[(np.abs(stats.zscore(states)) < 3.).all(axis=1)]
    temp_sample_df = pd.DataFrame(states)
    sample_df = temp_sample_df.melt()

    g = sns.displot(data=sample_df, x="value", row="variable", kind="kde", fill=1, color="blue",
                    height=2.5, aspect=3, facet_kws=dict(margin_titles=True), )

    count = 0
    for ax in g.axes.flat:
        ax.axvline(true_theta[count], color="red")
        ax.axvline(point_estimators[count], color="purple")
        count += 1
    g.fig.suptitle("Acceptance Ratio: " + str(accept_ratio) + ", Execution Time: " + str(execution_time))
    g.savefig(my_path + '/plots/SimpleNormal_mcmc_samples.png')

    # Simulate values from the true distribution and compare them to the values sampled by MCMC chain.
    num_values = int(len(sample_df.index))
    values_range_sigma = onp.linspace(0.5, 5., num=num_values)
    values_range_mu = onp.linspace(-2., 4.5, num=num_values)
    mu_1 = ybar1 - sd1 * np.array([scipy.stats.t.rvs(n - 1, size=number_of_chains * number_of_iters)]) * (n ** (-0.5))
    mu_2 = ybar2 - sd2 * np.array([scipy.stats.t.rvs(n - 1, size=number_of_chains * number_of_iters)]) * (n ** (-0.5))
    mu_3 = ybar3 - sd3 * np.array([scipy.stats.t.rvs(n - 1, size=number_of_chains * number_of_iters)]) * (n ** (-0.5))

    sigma_1 = sd1 ** 2. * np.array([scipy.stats.invgamma.rvs(a=(0.5 * (n - 1)), scale=(0.5 * (n - 1)),
                                                             size=number_of_chains * number_of_iters)])
    sigma_2 = sd2 ** 2. * np.array([scipy.stats.invgamma.rvs(a=(0.5 * (n - 1)), scale=(0.5 * (n - 1)),
                                                             size=number_of_chains * number_of_iters)])
    sigma_3 = sd3 ** 2. * np.array([scipy.stats.invgamma.rvs(a=(0.5 * (n - 1)), scale=(0.5 * (n - 1)),
                                                             size=number_of_chains * number_of_iters)])

    true_samples = np.concatenate((mu_1, mu_2, mu_3, sigma_1, sigma_2, sigma_3)).transpose()
    true_samples_df_temp = pd.DataFrame(true_samples)
    true_samples_df = true_samples_df_temp.melt()
    print("The true sampled data looks like:")
    print(true_samples_df.describe())
    print("The FHMC samples look like:")
    print(sample_df.describe())

    full_data = pd.concat((sample_df, true_samples_df))
    x = onp.array(["Sampled Distribution", "True Distribution"])
    labels = onp.repeat(x, [len(sample_df.index), len(true_samples_df.index)], axis=0)

    full_data['labels'] = labels
    nrows, ncols = full_data.shape
    full_data.index = range(nrows)

    g = sns.displot(data=full_data, x="value", row="variable", hue="labels", kind="kde",
                    height=2.5, aspect=3, facet_kws=dict(margin_titles=True), )
    count = 0
    for ax in g.axes.flat:
        ax.axvline(true_theta[count], color="red")
        count += 1
    g.fig.suptitle("Acceptance Ratio: " + str(accept_ratio) + ", Execution Time: " + str(execution_time))
    g.savefig(my_path + '/plots/SimpleNormal_mcmc_vs_truth2.png')


run_example()
graph_results()
