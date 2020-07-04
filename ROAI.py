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
def generate_instance_bernoulli(n_normal, n_outlier, mean, std, k):
    np.random.seed()
    if (mean + (k - 0.00001) * std > 1) or (mean - (k) * std < 0):
        print('adjust mean or std, not likely to generate bernoulli instances')

    y_normal = np.random.normal(mean, std, n_normal)

    for i in range(n_normal):
        if y_normal[i] > mean + (k - 0.00001) * std:
            y_normal[i] = mean + (k - 0.00001) * std
        # we clip to mean + (k - 0.00001) * std to avoid an arm with mean equal to the true threshold
        elif y_normal[i] < mean - (k) * std:
            y_normal[i] = mean - (k) * std

    outlier_start = 0.8
    outlier_end = 1
    y_outlier = np.random.uniform(outlier_start, outlier_end, n_outlier)
    print('outlier_set')
    print(y_outlier)

    instance = np.hstack((y_normal, y_outlier))

    return instance

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

def find_good_instance(instance, mean, std, k):

    # this function is trying to find instance such that there is no ambiguous about the outliers,
    # based on all three definitions: mean based, median based, and true outlier
    # this function could be relaxed later by adding tolerance to classification rules

    if_good_instance = 1
    threshold_true, threshold_median, threshold_mean = find_threshold(instance, mean, std, k)
    # make sure both thresholds are good
    s_outlier_true = []
    s_outlier_mean = []
    s_outlier_median = []

    for i in range(len(instance)):
        if instance[i] > threshold_true:
            s_outlier_true.append(i)
        if instance[i] > threshold_mean:
            s_outlier_mean.append(i)
        if instance[i] > threshold_median:
            s_outlier_median.append(i)

    if s_outlier_median != s_outlier_true or s_outlier_mean != s_outlier_true:
        if_good_instance = 0
    for i in instance:
        if (i == threshold_median) or (i == threshold_mean) or (i == threshold_true):
            if_good_instance = 0

    return if_good_instance





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





def single_sim(n_normal, n_outlier, n_select_list, mean, std, k_para, instance_type, sigma, delta, tol, pull_max, update_interval):

    np.random.seed()

    if_good_instance = 0
    while if_good_instance == 0:
        instance = generate_instance_bernoulli(n_normal, n_outlier, mean, std, k_para)
        if_good_instance = find_good_instance(instance, mean, std, k_para)

    threshold_true, threshold_median, threshold_mean = find_threshold(instance, mean, std, k_para)

    list_threshold = [threshold_true, threshold_median, threshold_mean]
    minimum_true = min(abs(instance - threshold_true))
    minimum_median = min(abs(instance - threshold_median))
    minimum_mean = min(abs(instance - threshold_mean))

    list_minimum_dist = [minimum_true, minimum_median, minimum_mean]
    list_error_multi = []
    list_error_spec_tol_multi = []
    list_error_spec_multi = []


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


    return list_error_multi, list_error_spec_tol_multi, list_error_spec_multi, list_threshold, list_minimum_dist

def multi_sim(n_parallel, n_process, mean, std, k_para, instance_type, sigma, delta, tol,
              n_normal, n_outlier, pull_max, update_interval, n_select_list):
    time_start = time.time()

    if_good_instance = 0
    while if_good_instance == 0:
        instance = generate_instance_bernoulli(n_normal, n_outlier, mean, std, k_para)
        if_good_instance = find_good_instance(instance, mean, std, k_para)

    threshold_true, threshold_median, threshold_mean = find_threshold(instance, mean, std, k_para)
    print('followings are only for one realization, just to show some statistics')
    print('threshold_true = {}'.format(threshold_true))
    print('threshold_median = {}'.format(threshold_median))
    print('threshold_mean = {}'.format(threshold_mean))


    single_sim_partial = partial(single_sim, n_normal, n_outlier, n_select_list, mean, std, k_para, instance_type,
                                 sigma, delta, tol, pull_max)

    pool = multiprocessing.Pool(processes = n_process)
    results = pool.map(single_sim_partial, list(map(int, update_interval * np.ones(n_parallel))))

    print('multi_sim got results!')

    # the order of the following sequences matters!!
    measures = ['error', 'error_spec_tol', 'error_spec']
    # error_spec calculate error with their own specific outlier threshold, which is the same as 'error' is this setting
    # error_spec_tol calculate error with some allowed tolerance, the result is similar to 'error'


    algs = ['ROAILUCB', 'ROAIElim', 'Random', 'WRR', 'RR']

    dict_error = dict(zip(algs, [[] for alg in algs]))
    dict_error_spec_tol = dict(zip(algs, [[] for alg in algs]))
    dict_error_method_specific = dict(zip(algs, [[] for alg in algs]))



    # orders need to match the previous one!
    dict_results = dict(zip(measures, [dict_error, dict_error_spec_tol, dict_error_method_specific]))

    dict_results_ave = copy.deepcopy(dict_results)
    dict_results_std = copy.deepcopy(dict_results)

    for i in range(n_parallel):
        for j in range(len(measures)):
            for k in range(len(algs)):
                dict_results[measures[j]][algs[k]].append(results[i][j][k])

    list_threshold = []
    list_minimum_dist = []
    for i in range(n_parallel):
        j = len(measures)
        # the output of list_threshold
        list_threshold.append(results[i][j])
        list_minimum_dist.append(results[i][j+1])
    list_threshold_ave = np.mean(list_threshold, axis=0)
    list_minimum_dist_ave = np.mean(list_minimum_dist, axis=0)



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

    x = list(range(update_interval, pull_max+1, update_interval))

    fig = plt.figure(1)

    for i in reversed(range(len(algs))):
        alg_ave = np.array(dict_results_ave[measures[0]][algs[i]])
        alg_std = std_adjust * np.array(dict_results_std[measures[0]][algs[i]])
        plt.errorbar(x, alg_ave, yerr=alg_std, label=algs[i], linewidth=2, errorevery=2)


    plt.xlabel('Number of pulls')
    plt.ylabel('Error rate')
    plt.legend(loc=0)
    plt.grid(alpha=0.75)
    plt.legend(loc='upper right')
    plt.xlim(0, pull_max)
    plt.ylim(0, 1)
    plt.savefig('synthetic_data_anytime.pdf')
    plt.show()
    plt.close(fig)




    # save data in the following file
    file = open('Synthetic_data.txt', 'w')
    file.write('{} - {}sigma - {}max - {}interval - {}n_parallel - {}n_normal - {}n_outlier - {}mean - {}std -{}instance_type\n'.\
              format(date.today(), sigma, pull_max, update_interval, n_parallel, n_normal, n_outlier, mean, std, instance_type))
    file.write('average threshold are as followings:\n')
    file.write('threshold_true, threshold_median, threshold_mean \n')
    file.write('threshold: {}\n'.format(list_threshold_ave))
    file.write('minimum distance: {}\n'.format(list_minimum_dist_ave))

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


def for_multi_sim():
    # std is to control the scale of the means of arms
    # while sigma is the subgaussian parameter for each arm/distribution if instance type is subgaussian

    instance_type = 'bernoulli'
    # WRR and RR only works for bounded distribution
    n_parallel = 500
    n_process = 100
    mean = 0.3
    std = 0.075
    # mean and std of normal (instance-generating) distribution that controls the mean of each arm
    k = 3
    sigma = 0.1
    # std of each arm if instance_type is normal, not used here due to Bernoulli
    delta = 0.05
    # confidence parameter

    tol = 0.5 * std
    # tolerance given for identification target, we output result both with and without tolerance
    pull_max = 10000
    update_interval = 200
    n_total = 105
    n_outlier = 5
    n_normal = n_total - n_outlier

    n_select_list = [n_total]
    # we test with all arms selected for the construction of the outlier threshold

    if instance_type == 'bernoulli':
        if (mean + (k-0.00001)*std > 1) or (mean - (k)*std < 0):
            # we clip to mean + (k - 0.00001) * std to avoid an arm with mean equal to the true threshold
            print('adjust mean or std, not likely to generate bernoulli instances')
            return 0

    multi_sim(n_parallel, n_process, mean, std, k, instance_type, sigma, delta, tol,
              n_normal, n_outlier, pull_max, update_interval, n_select_list)



if __name__ == '__main__':

    for_multi_sim()



