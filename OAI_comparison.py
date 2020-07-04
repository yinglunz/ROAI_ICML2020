import numpy as np
import math
import time
import multiprocessing
import copy
from functools import partial
from datetime import date
from astropy.stats import median_absolute_deviation

def generate_instance_bernoulli(n_normal, n_outlier, mean, std, k, outlier_start):
    # this is slightly different than the one in ROAI.py as we need to vary outlier_start
    np.random.seed()
    if (mean + (k - 0.00001) * std > 1) or (mean - k * std < 0):
        print('adjust mean or std, not likely to generate bernoulli instances')
    y_normal = np.random.normal(mean, std, n_normal)
    for i in range(n_normal):
        if y_normal[i] > mean + (k - 0.00001) * std:
            y_normal[i] = mean + (k - 0.00001) * std
            # we clip to mean + (k - 0.00001) * std to avoid an arm with mean equal to the true threshold
        elif y_normal[i] < mean - k * std:
            y_normal[i] = mean - k * std
    outlier_end = 1
    y_outlier = np.random.uniform(outlier_start, outlier_end, n_outlier)
    instance = np.hstack((y_normal, y_outlier))

    return instance

def find_threshold(instance, mean, std, k):
    # output the true threshold, the median-based threhsold and the mean-based threshold
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

    for i in instance:
        if (i == threshold_median) or (i == threshold_mean) or (i == threshold_true):
            if_good_instance = 0

    return if_good_instance




def single_calc_ROAI(index_select, instance, threshold_true, k_para, tol, delta):
    # calculate deviation and complexity of ROAI
    n = len(instance)
    n_select = len(index_select)
    if n_select % 2 == 1:
        start = int((n_select - 1) / 2)
        end = int((n_select + 1) / 2)
    else:
        start = int((n_select - 2) / 2)
        end = int((n_select + 2) / 2)
    ranking_select = np.argsort(instance[index_select])
    # s_l_select = index_select[ranking_select[:start]]
    s_m_select = index_select[ranking_select[start:end]]
    # s_u_select = index_select[ranking_select[end:]]
    median_select = sum(instance[i] for i in s_m_select)/(len(s_m_select))
    AD = np.zeros(n)
    for i in range(n):
        AD[i] = abs(median_select - instance[i])
    ranking_AD_select = np.argsort(AD[index_select])
    s_MAD_select = index_select[ranking_AD_select[start:end]]
    MAD_select = sum(AD[i] for i in s_MAD_select)/(len(s_MAD_select))
    threshold_select = median_select + 1.4826*k_para*MAD_select
    deviation = abs(threshold_select - threshold_true)
    Delta_theta = abs(instance - threshold_select)
    Delta_median = abs(instance - median_select)
    Delta_MAD = abs(AD - MAD_select)
    Delta_theta_star = min(Delta_theta)
    Delta_character = np.ones(n)
    for i in range(n):
        if i in index_select:
            Delta_character[i] = max(Delta_theta_star, min(Delta_theta[i], Delta_median[i], Delta_MAD[i]))
        else:
            Delta_character[i] = Delta_theta[i]

    complexity_Omega = (k_para**2) * sum(math.log(n*(k_para ** 2)/(delta * ((max(i, tol)) ** 2))) / ((max(i, tol)) ** 2) for i in Delta_character[index_select])
    other_index = list(set(range(n)).difference(set(index_select)))
    complexity_other = sum(math.log(n*(k_para ** 2)/(delta * ((max(i, tol)) ** 2))) / ((max(i, tol)) ** 2) for i in Delta_character[other_index])
    complexity = complexity_Omega + complexity_other


    return deviation, complexity



def single_calc(n_select_list, n_normal, n_outlier, mean, std, k_para, tol, outlier_start, delta):

    # calculate diff and complexity of median-based method and mean-based method

    np.random.seed()
    if_good_instance = 0
    while if_good_instance == 0:
        instance = generate_instance_bernoulli(n_normal, n_outlier, mean, std, k_para, outlier_start)
        if_good_instance = find_good_instance(instance, mean, std, k_para)

    threshold_true, threshold_median, threshold_mean = find_threshold(instance, mean, std, k_para)
    # print('threshold_true = {}'.format(threshold_true))
    # print('threshold_median = {}'.format(threshold_median))
    # print('threshold_mean = {}'.format(threshold_mean))

    deviation_RR = abs(threshold_true - threshold_mean)
    deviation_list = [deviation_RR, deviation_RR]
    n = len(instance)
    Delta_RR = abs(instance - threshold_mean)
    H1_RR = n / (max(min(Delta_RR), tol) ** 2)
    g_k = ((1 + (k_para * math.sqrt(n - 1))) ** 2) / n
    l_k = (np.sqrt(g_k) + np.sqrt((k_para ** 2) / (2 * np.log((np.pi ** 2 * n ** 3) / (6 * delta))))) ** 2
    H_RR = H1_RR * ((1 + np.sqrt(l_k)) ** 2)
    rho_star = (((n - 1) ** 2) / l_k) ** (1 / 3)
    # rho_star as suggested by zhuang et al 2017
    H2_WRR = sum( (1 / ((max(i, tol)) ** 2)) for i in Delta_RR)
    H_WRR = (H1_RR / rho_star + (rho_star - 1) * H2_WRR / rho_star) * ((1 + np.sqrt(l_k * rho_star)) ** 2)
    H_RR = H_RR * math.log((n* H_RR)/(delta))
    H_WRR = H_WRR * math.log((n*H_WRR)/delta)
    complexity_list = [H_RR, H_WRR]

    for n_select in n_select_list:
        index_select = np.random.choice(n, n_select, replace=False)
        deviation, complexity = single_calc_ROAI(index_select, instance, threshold_true, k_para, tol, delta)
        deviation_list.append(deviation)
        complexity_list.append(complexity)

    return deviation_list, complexity_list

def averaged_single_calc(n_select_list, n_parallel, n_process, n_normal, n_outlier, mean, std, k_para, tol, outlier_start, delta):
    # calucate the avareged/median results of diff and complexity

    single_sim_partial = partial(single_calc, n_select_list, n_normal, n_outlier, mean, std, k_para, tol, outlier_start)
    pool = multiprocessing.Pool(processes = n_process)
    results = pool.map(single_sim_partial, list(map(float, delta * np.ones(n_parallel))))
    print('multi_sim got results!')
    measures = ['deviation', 'complexity']
    algs = ['RR', 'WRR']
    algs_ROAI = ['ROAI_{}'.format(i) for i in n_select_list]
    algs = algs + algs_ROAI
    print('algs={}'.format(algs))
    dict_deviation = dict(zip(algs, [[] for i in algs]))
    dict_complexity = dict(zip(algs, [[] for i in algs]))
    dict_results = dict(zip(measures, [dict_deviation, dict_complexity]))
    dict_results_ave = copy.deepcopy(dict_results)
    dict_results_median = copy.deepcopy(dict_results)
    dict_results_std = copy.deepcopy(dict_results)
    dict_results_mad = copy.deepcopy(dict_results)
    for i in range(n_parallel):
        for j in range(len(measures)):
            for k in range(len(algs)):
                dict_results[measures[j]][algs[k]].append(results[i][j][k])
    for measure in measures:
        for alg in algs:
            dict_results_ave[measure][alg] = np.mean(dict_results[measure][alg], axis=0)
            dict_results_median[measure][alg] = np.median(dict_results[measure][alg], axis=0)
            dict_results_std[measure][alg] = np.std(dict_results[measure][alg], axis=0)
            dict_results_mad[measure][alg] = median_absolute_deviation(dict_results[measure][alg], axis=0)
    deviation_mean = []
    deviation_std = []

    complexity_median = []
    complexity_MAD = []
    for alg in algs:
        deviation_mean.append(dict_results_ave['deviation'][alg])
        deviation_std.append(dict_results_std['deviation'][alg])
        complexity_median.append(dict_results_median['complexity'][alg])
        complexity_MAD.append(dict_results_mad['complexity'][alg])

    return deviation_mean, deviation_std, complexity_median, complexity_MAD


def multi_calc(contamination_level_list, n_total):
    # compare deviation with varying contamination level
    time_start = time.time()
    n_parallel = 500
    n_process = 100
    mean = 0.3
    std = 0.075
    k_para = 3
    tol = 0
    delta = 0.05
    instance_type = 'bernoulli'

    deviation_mean_multi_list = []
    deviation_std_multi_list = []

    for contamination_level in contamination_level_list:
        print('contamination level = {}'.format(contamination_level))
        n_outlier = int(np.floor(n_total * contamination_level))
        n_normal = n_total - n_outlier
        # select a subset of arm
        n_select_list = [int(2 * np.ceil(n_total * contamination_level) + 1),
                           int(4 * np.ceil(n_total * contamination_level) + 1), n_total]
        print('n_select_list={}'.format(n_select_list))
        for i in range(len(n_select_list)):
            if n_select_list[i] > n_total:
                n_select_list[i] = n_total
            if n_select_list[i] < 15:
                n_select_list[i] = 15
        print('refined n_select_list={}'.format(n_select_list))
        deviation_mean, deviation_std, complexity_median, complexity_MAD = averaged_single_calc(
            n_select_list, n_parallel, n_process, n_normal, n_outlier,mean, std, k_para, tol, 0.7, delta)
        # starting point of the uniform distribution is 0.7 with different contamination levels
        deviation_mean_multi_list.append(deviation_mean)
        deviation_std_multi_list.append(deviation_std)


    time_end = time.time()
    time_total = time_end - time_start
    deviation_mean_array = np.array(deviation_mean_multi_list)
    deviation_std_array = np.array(deviation_std_multi_list)

    measures = ['deviation']
    algs = ['RR', 'WRR', '2ne', '4ne', 'n']

    file = open('deviation wrt varying contamination level.txt', 'w')
    file.write('date - {}\n'.format(date.today()))
    file.write('total time - {}\n'.format(time_total))
    file.write('{}n_total\n'.format(n_total))
    file.write('outlier range [0.7, 1]\n')
    file.write('{}n_parallel - {}n_process - {}mean - {}std - {}k_para - {}delta\n'.format(
        n_parallel, n_process, mean, std, k_para, delta))
    file.write('measures:{}\n'.format(measures))
    file.write('algs:{}\n'.format(algs))

    file.write('varying contamination level = {}\n'.format(contamination_level_list))

    file.write('----deviation mean is as following----\n')
    for i in range(len(algs)):
        file.write('alg: {}\n'.format(algs[i]))
        file.write('{}\n'.format(deviation_mean_array[:, i]))

    file.write('----deviation std is as following----\n')
    for i in range(len(algs)):
        file.write('alg: {}\n'.format(algs[i]))
        file.write('{}\n'.format(deviation_std_array[:, i]))




def multi_calc_outlier_start(outlier_start_list, n_total):
    # compare complexity with varying starting point of the uniform distribution
    time_start = time.time()
    n_parallel = 500
    n_process = 100
    mean = 0.3
    std = 0.075
    k_para = 3
    tol = 0
    # we give 0 tolerance and compare the median complexity
    delta = 0.05
    instance_type = 'bernoulli'

    complexity_median_multi_list = []
    complexity_MAD_multi_list = []
    for outlier_start in outlier_start_list:
        print('outlier start = {}'.format(outlier_start))
        n_outlier = 10
        n_normal = n_total - n_outlier
        contamination_level = n_outlier/n_total
        # select a subset of arm
        n_select_list = [int(2 * np.ceil(n_total * contamination_level) + 1),
                           int(4 * np.ceil(n_total * contamination_level) + 1), n_total]
        print('n_select_list={}'.format(n_select_list))

        for i in range(len(n_select_list)):
            if n_select_list[i] > n_total:
                n_select_list[i] = n_total
            if n_select_list[i] < 15:
                n_select_list[i] = 15
        print('refined n_select_list={}'.format(n_select_list))
        deviation_mean, deviation_std, complexity_median, complexity_MAD = averaged_single_calc(
            n_select_list, n_parallel, n_process, n_normal, n_outlier,mean, std, k_para, tol, outlier_start, delta)

        complexity_median_multi_list.append(complexity_median)
        complexity_MAD_multi_list.append(complexity_MAD)

    time_end = time.time()
    time_total = time_end - time_start

    complexity_median_array = np.array(complexity_median_multi_list)
    complexity_MAD_array = np.array(complexity_MAD_multi_list)

    measures = ['complexity']
    algs = ['RR', 'WRR', '2ne', '4ne', 'n']
    file = open('complexity wrt varying starting point of Uniform distribution.txt', 'w')
    file.write('date - {}\n'.format(date.today()))
    file.write('total time - {}\n'.format(time_total))
    file.write('{}n_total {}n_outlier\n'.format(n_total, 10))
    file.write('{}n_parallel - {}n_process - {}mean - {}std - {}k_para - {}delta - {}instance_type\n'.format(
        n_parallel, n_process, mean, std, k_para, delta, instance_type))
    file.write('measures:{}\n'.format(measures))
    file.write('algs:{}\n'.format(algs))

    file.write('varying starting point is {}\n'.format(outlier_start_list))
    file.write('----complexity median as followings----\n')
    for i in range(len(algs)):
        file.write('alg: {}\n'.format(algs[i]))
        file.write('{}\n'.format(complexity_median_array[:,i]))
    file.write('----complexity MAD as followings----\n')
    for i in range(len(algs)):
        file.write('alg: {}\n'.format(algs[i]))
        file.write('{}\n'.format(complexity_MAD_array[:,i]))

if __name__ == '__main__':

    n_total = 105

    contamination_level = 1
    # set contamination_level = 1 to test deviation wrt contamination level, otherwise to test sample complexity wrt hardness


    if contamination_level == 1:
        contamination_level_list = np.linspace(0, 0.2, 20)
        multi_calc(contamination_level_list, n_total)
    else:
        outlier_start_list = np.linspace(0.6, 0.9, 20)
        multi_calc_outlier_start(outlier_start_list, n_total)





