import numpy as np
from math import sqrt
import time
import multiprocessing
import copy
from functools import partial
from datetime import date
import matplotlib.pyplot as plt
import matplotlib

from ROAI_class import ROAI, RANDOM, RR, WRR


# this function is used to generate the instance


def single_run_RR(instance, mean, std, k, sigma, delta, tol, pull_max, update_interval):
    list_error = []
    list_error_spec_tol = []
    list_error_spec = []


    alg_obj = RR(instance, mean, std, k, sigma, delta, tol)
    alg_obj.initialization()

    while alg_obj.t <= pull_max:
        alg_obj.update()
        if alg_obj.t % update_interval == 0:

            error_at, error_spec_tol_at, error_spec_at = alg_obj.compute_error()
            list_error.append(error_at)
            list_error_spec_tol.append(error_spec_tol_at)
            list_error_spec.append(error_spec_at)


    return list_error, list_error_spec_tol, list_error_spec


def single_run_WRR(instance, mean, std, k, sigma, delta, tol, pull_max, update_interval):
    list_error = []
    list_error_spec_tol = []
    list_error_spec = []

    alg_obj = WRR(instance, mean, std, k, sigma, delta, tol)
    alg_obj.initialization()
    while alg_obj.t <= pull_max:
        regular_arm = alg_obj.index_pull
        alg_obj.update_regular()
        if alg_obj.t % update_interval == 0:
            error_at, error_spec_tol_at, error_spec_at = alg_obj.compute_error()
            list_error.append(error_at)
            list_error_spec_tol.append(error_spec_tol_at)
            list_error_spec.append(error_spec_at)


        while alg_obj.t <= pull_max and regular_arm in alg_obj.active_set and alg_obj.pulls[regular_arm] < alg_obj.threshold_pulls[regular_arm]:
            alg_obj.update_additional(regular_arm)
            if alg_obj.t % update_interval == 0:

                error_at, error_spec_tol_at, error_spec_at = alg_obj.compute_error()
                list_error.append(error_at)
                list_error_spec_tol.append(error_spec_tol_at)
                list_error_spec.append(error_spec_at)


    return list_error, list_error_spec_tol, list_error_spec


def single_run_random(instance, n_select, mean, std, k, instance_type, sigma, delta, tol, pull_max, update_interval):
    list_error = []
    list_error_spec_tol = []
    list_error_spec = []


    # select all arms in the rondom sampling strategy

    alg_obj = RANDOM(instance, n_select, mean, std, k, instance_type, sigma, delta, tol)
    alg_obj.initialization()

    while alg_obj.t <= pull_max:
        alg_obj.update()
        if alg_obj.t % update_interval == 0:
            error_at, error_spec_tol, error_spec_at = alg_obj.compute_error()
            list_error.append(error_at)
            list_error_spec_tol.append(error_spec_tol)
            list_error_spec.append(error_spec_at)


    return list_error, list_error_spec_tol, list_error_spec



def single_run_ROAI_elimi(instance, n_select, mean, std, k, instance_type, sigma, delta, tol, pull_max, update_interval):
    list_error = []
    list_error_spec_tol = []
    list_error_spec = []

    alg_obj = ROAI(instance, n_select, mean, std, k, instance_type, sigma, delta, tol)
    alg_obj.initialization_elimi()
    while alg_obj.t <= pull_max:
        alg_obj.update_elimi()

        if alg_obj.t % update_interval == 0:
            error_at, error_spec_tol, error_spec_at = alg_obj.compute_error()
            list_error.append(error_at)
            list_error_spec_tol.append(error_spec_tol)
            list_error_spec.append(error_spec_at)


    return list_error, list_error_spec_tol, list_error_spec

def single_run_ROAI_lucb(instance, n_select, mean, std, k, instance_type, sigma, delta, tol, pull_max, update_interval):
    list_error = []
    list_error_spec_tol = []
    list_error_spec = []


    alg_obj = ROAI(instance, n_select, mean, std, k, instance_type, sigma, delta, tol)
    alg_obj.initialization_lucb()

    while alg_obj.t <= pull_max:
        alg_obj.update_lucb()

        if alg_obj.t % update_interval == 0:
            error_at, error_spec_tol, error_spec_at = alg_obj.compute_error()
            list_error.append(error_at)
            list_error_spec_tol.append(error_spec_tol)
            list_error_spec.append(error_spec_at)


    return list_error, list_error_spec_tol, list_error_spec


def find_threshold(instance, mean, std, k):
    n = len(instance)
    if n % 2 == 1:
        start = int((n - 1) / 2)
        end = int((n + 1) / 2)
    else:
        start = int((n - 2) / 2)
        end = int((n + 2) / 2)

    ranking = np.argsort(instance)
    s_median = ranking[start:end]
    median = sum(instance[i] for i in s_median) / len(s_median)
    AD = np.zeros(n)
    for i in range(n):
        AD[i] = abs(instance[i] - median)
    ranking_AD = np.argsort(AD)
    s_median_AD = ranking_AD[start:end]
    median_AD = sum(AD[i] for i in s_median_AD) / len(s_median_AD)
    threshold_median = median + k * 1.4826 * median_AD
    threshold_mean = np.mean(instance) + k * np.std(instance)
    threshold_true = mean + k * std

    return threshold_true, threshold_median, threshold_mean



def single_sim(n_select_list, mean, std, k_para, instance_type, sigma, delta, tol, pull_max, instance, update_interval):

    np.random.seed()

    list_error_spec_tol_multi = []
    list_error_spec_multi = []
    list_error_multi = []


    for n_select in n_select_list:
        list_error, list_error_spec_tol, list_error_spec = single_run_ROAI_lucb \
            (instance, n_select, mean, std, k_para, instance_type, sigma, delta, tol, pull_max, update_interval)
        list_error_multi.append(list_error)
        list_error_spec_tol_multi.append(list_error_spec_tol)
        list_error_spec_multi.append(list_error_spec)


    print('finished ROAI_lucb')

    for n_select in n_select_list:
        list_error, list_error_spec_tol, list_error_spec = single_run_ROAI_elimi \
            (instance, n_select, mean, std, k_para, instance_type, sigma, delta, tol, pull_max, update_interval)
        list_error_multi.append(list_error)
        list_error_spec_tol_multi.append(list_error_spec_tol)
        list_error_spec_multi.append(list_error_spec)


    print('finished ROAI_elimi')
    for n_select in n_select_list:
        list_error, list_error_spec_tol, list_error_spec = single_run_random \
            (instance, n_select, mean, std, k_para, instance_type, sigma, delta, tol, pull_max, update_interval)
        list_error_multi.append(list_error)
        list_error_spec_tol_multi.append(list_error_spec_tol)
        list_error_spec_multi.append(list_error_spec)



    print('finished Random')


    list_error, list_error_spec_tol, list_error_spec = single_run_WRR \
        (instance, mean, std, k_para, sigma, delta, tol, pull_max, update_interval)
    list_error_multi.append(list_error)
    list_error_spec_tol_multi.append(list_error_spec_tol)
    list_error_spec_multi.append(list_error_spec)


    print('finished WRR')

    list_error, list_error_spec_tol, list_error_spec = single_run_RR\
        (instance, mean, std, k_para, sigma, delta, tol, pull_max, update_interval)
    list_error_multi.append(list_error)
    list_error_spec_tol_multi.append(list_error_spec_tol)
    list_error_spec_multi.append(list_error_spec)


    print('finished RR')



    return list_error_multi, list_error_spec_tol_multi, list_error_spec_multi

def multi_sim(n_parallel, n_process, mean, std, k_para, instance_type, sigma, delta, tol,
              pull_max, update_interval, n_select_list, instance):
    time_start = time.time()

    threshold_true, threshold_median, threshold_mean = find_threshold(instance, mean, std, k_para)

    list_threshold = [threshold_true, threshold_median, threshold_mean]

    minimum_true = min(abs(instance - threshold_true))
    minimum_median = min(abs(instance - threshold_median))
    minimum_mean = min(abs(instance - threshold_mean))

    list_minimum_dist = [minimum_true, minimum_median, minimum_mean]


    single_sim_partial = partial(single_sim, n_select_list, mean, std, k_para, instance_type,
                                 sigma, delta, tol, pull_max, instance)

    pool = multiprocessing.Pool(processes = n_process)
    results = pool.map(single_sim_partial, list(map(int, update_interval * np.ones(n_parallel))))

    print('multi_sim got results!')

    # the order of the following sequences matters!!
    measures = ['error', 'error_spec_tol', 'error_spec']
    # error_spec calculate error with their own specific outlier threshold, which is the same as 'error' is this setting
    # error_spec_tol calculate error with some allowed tolerance, the result is similar to 'error'


    algs = ['ROAILUCB', 'ROAIElim', 'Random', 'WRR', 'RR']

    dict_error_spec_tol = dict(zip(algs, [[] for alg in algs]))
    dict_error_method_specific = dict(zip(algs, [[] for alg in algs]))
    dict_error = dict(zip(algs, [[] for alg in algs]))



    # orders need to match the previous one!
    dict_results = dict(zip(measures, [dict_error_spec_tol, dict_error_method_specific, dict_error]))

    dict_results_ave = copy.deepcopy(dict_results)
    dict_results_std = copy.deepcopy(dict_results)

    for i in range(n_parallel):
        for j in range(len(measures)):
            for k in range(len(algs)):
                dict_results[measures[j]][algs[k]].append(results[i][j][k])



    for measure in measures:
        for alg in algs:
            dict_results_ave[measure][alg] = np.mean(dict_results[measure][alg], axis=0)
            dict_results_std[measure][alg] = np.std(dict_results[measure][alg], axis=0)

    print('---- final average results ----')
    print(dict_results_ave)

    time_end = time.time()
    print('total time spent', time_end - time_start)

    # plot figures
    matplotlib.rcParams.update({'font.size': 12})
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    std_adjust = 2/sqrt(500)

    x = list(range(update_interval, pull_max + 1, update_interval))

    fig = plt.figure(1)

    for i in reversed(range(len(algs))):
        alg_ave = np.array(dict_results_ave[measures[0]][algs[i]])
        alg_std = std_adjust * np.array(dict_results_std[measures[0]][algs[i]])
        plt.errorbar(x, alg_ave, yerr=alg_std, label=algs[i], linewidth=2, errorevery=2)

    plt.xlabel('Number of pulls')
    plt.ylabel('Error rate')
    plt.grid(alpha=0.75)
    plt.legend(loc='upper right')
    plt.xlim(0, pull_max)
    plt.ylim(0, 1)
    plt.savefig('realdata_anytime.pdf')
    plt.show()
    plt.close(fig)




    # save data in the following file

    file = open('Real_data.txt', 'w')
    file.write('{} - {}sigma - {}max - {}interval - {}n_parallel - {}mean - {}std -{}instance_type\n'.\
              format(date.today(), sigma, pull_max, update_interval, n_parallel, mean, std, instance_type))
    file.write('approximate threshold (from one realization) are as followings:\n')
    file.write('threshold_true, threshold_median, threshold_mean \n')
    file.write('threshold: {}\n'.format(list_threshold))
    file.write('minimum distance: {}\n'.format(list_minimum_dist))

    file.write('k={} - delta={}\n'.format(k_para, delta))
    file.write('total time spent = {}\n'.format(time_end - time_start))

    file.write('measures: {}'.format(measures))
    file.write('algs: {}\n'.format(algs))

    for measure in measures:
        for alg in algs:
            file.write('measure:{}, alg:{}, ave:\n'.format(measure, alg))
            file.write('{}\n'.format(dict_results_ave[measure][alg]))

    for measure in measures:
        for alg in algs:
            file.write('measure:{}, alg:{}, std:\n'.format(measure, alg))
            file.write('{}\n'.format(dict_results_std[measure][alg]))


def for_multi_sim(y_normal, y_outlier):
    # std is to control the scale of the means of arms
    # while sigma is the subgaussian parameter for each arm/distribution if instance type is subgaussian

    instance = np.hstack((y_outlier, y_normal))
    instance_type = 'bernoulli'
    # WRR and RR only works for bounded distribution

    n_parallel = 500
    n_process = 100
    mean = 0.5
    std = 0.1
    # mean and std are not used anymore as we are working with real data
    # the only requirement is that 0.5 + k * 0.1 \in [0.57, 0.84] (approximately)
    # so such the true_threshold constructed could correctly identify the subset of outliers
    k = 3
    sigma = 0.1
    # std of each arm if instance_type is normal, not used here due to Bernoulli
    delta = 0.05
    # confidence parameter

    tol = 0.5 * std
    # tolerance given for identification target, we output result both with and without tolerance

    pull_max = 10000
    update_interval = 200

    n_total = len(instance)

    n_select_list = [n_total]
    # we test with all arms selected for the construction of the outlier threshold

    multi_sim(n_parallel, n_process, mean, std, k, instance_type, sigma, delta, tol,
              pull_max, update_interval, n_select_list, instance)


if __name__ == '__main__':

    # First, get dataset `wine.mat` from http://odds.cs.stonybrook.edu/wine-dataset/. Next, preprocess the dataset based on the experiment description (remove 6 arms close to the threshold). Then, input the preprocessed means of normal and outlier arms into `y_normal` and `y_outlier`.

    y_normal = np.array([])

    y_outlier = np.array([])

    if len(y_normal) == 0 or len(y_outlier) == 0:
        raise Exception('input the preprocessed means of normal and outlier arms')

    instance = np.hstack((y_outlier, y_normal))

    for_multi_sim(y_normal, y_outlier)





