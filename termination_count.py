import numpy as np
from math import sqrt, log
import time
import multiprocessing
from functools import partial
from datetime import date


from ROAI_class import ROAI, RANDOM, RR, WRR


def generate_instance(outlier_start):
    # generate instance
    y_normal = np.linspace(0, 2, num=15)
    y_outlier = np.linspace(outlier_start, outlier_start + 0.2, 2)
    instance = np.hstack((y_normal, y_outlier))
    return instance


def single_calc_ROAI(index_select, instance, k_para, tol, delta):

    n = len(instance)
    n_select = len(index_select)

    if n_select % 2 == 1:
        start = int((n_select - 1) / 2)
        end = int((n_select + 1) / 2)
    else:
        start = int((n_select - 2) / 2)
        end = int((n_select + 2) / 2)

    ranking_select = np.argsort(instance[index_select])
    s_l_select = index_select[ranking_select[:start]]
    s_m_select = index_select[ranking_select[start:end]]
    s_u_select = index_select[ranking_select[end:]]

    median_select = sum(instance[i] for i in s_m_select)/(len(s_m_select))

    AD = np.zeros(n)

    for i in range(n):
        AD[i] = abs(median_select - instance[i])

    ranking_AD_select = np.argsort(AD[index_select])

    s_MAD_select = index_select[ranking_AD_select[start:end]]
    MAD_select = sum(AD[i] for i in s_MAD_select)/(len(s_MAD_select))

    threshold_select = median_select + 1.4826*k_para*MAD_select


    Delta_theta = abs(instance - threshold_select)
    Delta_median = abs(instance - median_select)
    Delta_MAD = abs(AD - MAD_select)

    Delta_theta_star = min(Delta_theta)
    print('Delta_theta_star = {}'.format(Delta_theta_star))

    Delta_character = np.ones(n)
    for i in range(n):
        if i in index_select:
            Delta_character[i] = max(Delta_theta_star, min(Delta_theta[i], Delta_median[i], Delta_MAD[i]))
        else:
            Delta_character[i] = max(Delta_theta_star, Delta_theta[i])

    complexity = sum(log(n*(k_para**2)/(delta * ((max(i, tol)) ** 2))) / ((max(i, tol)) ** 2) for i in Delta_character)
    complexity = complexity * (k_para**2)

    return complexity


def single_sim(instance, mean, std, k_para, instance_type, sigma, delta, tol, pull_max):

    np.random.seed()

    n_select = len(instance)
    ################

    alg_obj_lucb = ROAI(instance, n_select, mean, std, k_para, instance_type, sigma, delta, tol)
    alg_obj_lucb.initialization_lucb()
    s_active_len_lucb = len(instance)

    while alg_obj_lucb.t < pull_max and s_active_len_lucb != 0:
        if alg_obj_lucb.t % 2000 == 0:
            print('LUCB: s_active_len = {}, t = {}'.format(alg_obj_lucb.s_active_len, alg_obj_lucb.t))
        alg_obj_lucb.update_lucb()
        s_active_len_lucb = alg_obj_lucb.s_active_len

    count_lucb = alg_obj_lucb.t
    #################

    alg_obj_elim = ROAI(instance, n_select, mean, std, k_para, instance_type, sigma, delta, tol)
    alg_obj_elim.initialization_elimi()
    s_active_len_elim = len(instance)

    while alg_obj_elim.t < pull_max and s_active_len_elim != 0:
        if alg_obj_elim.t % 2000 == 0:
            print('ELIM: s_active_len = {}, t = {}'.format(alg_obj_elim.s_active_len, alg_obj_elim.t))
        alg_obj_elim.update_elimi()
        s_active_len_elim = alg_obj_elim.s_active_len

    count_elim = alg_obj_elim.t
    #################

    alg_obj_random = RANDOM(instance, n_select, mean, std, k_para, instance_type, sigma, delta, tol)
    alg_obj_random.initialization()
    s_active_len_random = len(instance)

    while alg_obj_random.t < pull_max and s_active_len_random != 0:
        if alg_obj_random.t % 2000 == 0:
            print('RANDOM: s_active_len = {}, t = {}'.format(alg_obj_random.s_active_len, alg_obj_random.t))
        alg_obj_random.update()
        s_active_len_random = alg_obj_random.s_active_len

    count_random = alg_obj_random.t
    ################

    return count_lucb, count_elim, count_random

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




def multi_sim(n_parallel, n_process, mean, std, k_para, instance_type, sigma, delta, tol,
             pull_max, outlier_start_adding):

    np.random.seed()


    instance = generate_instance(4)
    # the last two outlier arms won't affect threshold_median due to its robustness

    threshold_true, threshold_median, threshold_mean = find_threshold(instance, mean, std, k_para)
    print('threshold_median = {}'.format(threshold_median))
    outlier_start = outlier_start_adding + threshold_median
    print('outlier_start = {}'.format(outlier_start))


    instance = generate_instance(outlier_start)

    threshold_true, threshold_median, threshold_mean = find_threshold(instance, mean, std, k_para)

    print('threshold_median = {}'.format(threshold_median))

    index_select = np.arange(len(instance))
    complexity = single_calc_ROAI(index_select, instance, k_para, tol, delta)
    print('complexity = {}'.format(complexity))


    single_sim_partial = partial(single_sim, instance, mean, std, k_para, instance_type, sigma, delta, tol)

    pool = multiprocessing.Pool(processes = n_process)
    results = pool.map(single_sim_partial, list(map(int, pull_max * np.ones(n_parallel))))

    list_count_lucb = []
    list_count_elim = []
    list_count_random = []

    for i in range(n_parallel):
        list_count_lucb.append(results[i][0])
        list_count_elim.append(results[i][1])
        list_count_random.append(results[i][2])

    count_lucb_ave = np.mean(list_count_lucb)
    count_lucb_std = np.std(list_count_lucb)

    count_elim_ave = np.mean(list_count_elim)
    count_elim_std = np.std(list_count_elim)

    count_random_ave = np.mean(list_count_random)
    count_random_std = np.std(list_count_random)

    count_ave = [complexity, count_lucb_ave, count_elim_ave, count_random_ave]
    print('count_ave:{}'.format(count_ave))

    count_std = [0, count_lucb_std, count_elim_std, count_random_std]
    print('count_std:{}'.format(count_std))

    return count_ave, count_std


def for_multi_sim(outlier_start_adding_list):
    # std is to control the scale of the means of arms
    # while sigma is the subgaussian parameter for each arm/distribution if instance type is subgaussian
    time_start = time.time()
    np.random.seed()
    n_parallel = 500
    n_process = 100
    mean = 0.3
    std = 0.1
    # mean and std are not used anymore as we are working on a different data-generation method
    k_para = 2
    # set k_para = 2 to get the desired \Delta^{\theta}_{*}


    sigma = 0.5
    delta = 0.05
    # confidence parameter
    tol = 0
    # tolerance = 0
    pull_max = 1000000
    instance_type = 'gaussian'

    list_count_ave = []
    list_count_std = []

    for outlier_start_adding in outlier_start_adding_list:
        count_ave, count_std = multi_sim(n_parallel, n_process, mean, std, k_para, instance_type, sigma, delta, tol,
             pull_max, outlier_start_adding)
        list_count_ave.append(count_ave)
        list_count_std.append(count_std)

    time_end = time.time()
    time_total = time_end - time_start

    print(list_count_ave, list_count_std)

    count_ave_array = np.array(list_count_ave)
    count_std_array = np.array(list_count_std)

    theoretical_complexity = count_ave_array[:, 0]
    algs = ['ROAILUCB', 'ROAIElim', 'Random']
    result_ave = dict(zip(algs, [[] for alg in algs]))
    result_std = dict(zip(algs, [[] for alg in algs]))

    for i in range(len(algs)):
        result_ave[algs[i]] = count_ave_array[:, i+1]
        result_std[algs[i]] = count_std_array[:, i+1]


    # save data
    file = open('Termination_count.txt', 'w')
    file.write('{} - {}k - {}max - {}n_parallel - {}instance_type .txt\n'. \
               format(date.today(), k_para, pull_max, n_parallel, instance_type))
    file.write('y_normal = np.linspace(0, 2, num = 15)\n')
    file.write('y_outlier = np.linspace(outlier_start, outlier_start + 0.2, 2)\n')
    file.write('total time = {}\n'.format(time_total))
    file.write('outlier start list ={}\n'.format(outlier_start_adding_list))

    file.write('----theoretical complexity----ave \n')
    file.write('{}\n'.format(theoretical_complexity))

    for i in range(len(algs)):
        file.write('----{}----ave then std\n'.format(algs[i]))
        file.write('{}\n'.format(result_ave[algs[i]]))
        file.write('{}\n'.format(result_std[algs[i]]))



if __name__ == '__main__':


    outlier_start_adding_list = [0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.275, 0.25, 0.225, 0.2]

    for_multi_sim(outlier_start_adding_list)

