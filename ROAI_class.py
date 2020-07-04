import numpy as np
import math




def get_reward(instance, arm, sigma, instance_type):

    if instance_type == 'bernoulli':
        if np.random.random() < instance[arm]:
            return 1
        else:
            return 0
    else:
        return np.random.normal(instance[arm], sigma)


class RR:
    def __init__(self, instance, mean, std, k, sigma, delta, tol):
        self.instance = instance
        self.n = len(instance)
        n = self.n
        self.mean = mean
        self.std = std
        self.k = k
        # this k denotes the original one
        self.threshold_true = self.mean + self.k * self.std
        self.outlier_set_true = []
        self.instance_type = 'bernoulli'
        self.sigma = sigma
        self.delta = delta
        self.tol = tol

        self.t = 0
        self.active_set = []
        self.wins = np.zeros(n)
        self.pulls = np.zeros(n)
        self.rewards = np.zeros(n)
        # rewards here represents the empirical mean of each arm
        self.ucbs = np.ones(n)
        self.lcbs = np.zeros(n)

        self.index_pull = 0

        self.threshold_spec = 0
        self.outlier_set_spec = []
        self.outlier_set_spec_sub = []
        self.outlier_set_spec_sup = []

        self.threshold_at = 0
        self.threshold_lcb = 0
        self.threshold_ucb = 1
        self.outlier_set_at = []

    def compute_ci_hoeffding(self, arm):
        log_term = math.log((np.pi**2 * (self.n+1) * (self.pulls[arm] ** 2)) / (3 * self.delta))
        return math.sqrt(log_term / (2 * self.pulls[arm]))
    # note that they need a (n+1) term rather than a (n) term

    def compute_ci_threshold(self):
        log_term = math.log(((math.pi ** 2) * self.n * (self.t ** 2)) / (3 * self.delta))
        sum_inverse = 0
        for arm in range(self.n):
            sum_inverse += 1 / self.pulls[arm]
        harmonic_mean = self.n / sum_inverse
        return math.sqrt((log_term * self.l_k) / (2 * harmonic_mean))

    def initialization(self):

        g_k = ((1 + (self.k * math.sqrt(self.n - 1))) ** 2) / self.n
        self.l_k = (np.sqrt(g_k) +  np.sqrt(self.k**2/(2* np.log((np.pi**2 * self.n**3)/(6*self.delta)))))**2
        # compute true set of outliers
        self.threshold_spec = np.mean(self.instance) + self.k * np.std(self.instance)
        for arm in range(self.n):
            if self.instance[arm] > self.threshold_spec - self.tol:
                self.outlier_set_spec_sup.append(arm)
                if self.instance[arm] > self.threshold_spec:
                    self.outlier_set_spec.append(arm)
                    if self.instance[arm] > self.threshold_spec + self.tol:
                        self.outlier_set_spec_sub.append(arm)
            if self.instance[arm] > self.threshold_true:
                self.outlier_set_true.append(arm)
        # pull each arm once
        for arm in range(self.n):
            rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
            self.t += 1
            self.wins[arm] += rwd
            self.pulls[arm] += 1
            self.rewards[arm] = self.wins[arm] / self.pulls[arm]
            beta_tilde = self.compute_ci_hoeffding(arm)
            self.ucbs[arm] = self.rewards[arm] + beta_tilde
            self.lcbs[arm] = self.rewards[arm] - beta_tilde
        beta_tilde_threshold = self.compute_ci_threshold()
        self.threshold_at = np.mean(self.rewards) + self.k * np.std(self.rewards)
        self.threshold_ucb = self.threshold_at + beta_tilde_threshold
        self.threshold_lcb = self.threshold_at - beta_tilde_threshold
        self.outlier_set_at = []
        for arm in range(self.n):
            if self.rewards[arm] > self.threshold_at:
                self.outlier_set_at.append(arm)
            if (self.rewards[arm] > self.threshold_at and self.lcbs[arm] < self.threshold_ucb) or \
                    (self.rewards[arm] <= self.threshold_at and self.ucbs[arm] > self.threshold_lcb):
                self.active_set.append(arm)


    def update(self):
        arm = self.index_pull
        rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
        self.t += 1
        self.wins[arm] += rwd
        self.pulls[arm] += 1
        self.rewards[arm] = self.wins[arm] / self.pulls[arm]
        beta_tilde = self.compute_ci_hoeffding(arm)
        self.ucbs[arm] = self.rewards[arm] + beta_tilde
        self.lcbs[arm] = self.rewards[arm] - beta_tilde
        beta_tilde_threshold = self.compute_ci_threshold()
        self.threshold_at = np.mean(self.rewards) + self.k * np.std(self.rewards)
        self.threshold_ucb = self.threshold_at + beta_tilde_threshold
        self.threshold_lcb = self.threshold_at - beta_tilde_threshold
        self.outlier_set_at = []
        for i in range(self.n):
            if self.rewards[i] > self.threshold_at:
                self.outlier_set_at.append(i)
        for arm in self.active_set:
            if (self.rewards[arm] > self.threshold_at and self.threshold_ucb <= self.lcbs[arm]) or \
                    (self.rewards[arm] <= self.threshold_at and self.ucbs[arm] <= self.threshold_lcb):
                self.active_set.remove(arm)
        self.index_pull = (self.index_pull + 1) % self.n

    def compute_error(self):
        error_spec_tol = 1
        error_spec = 1
        if (set(self.outlier_set_at).issubset(set(self.outlier_set_spec_sup))) and \
            (set(self.outlier_set_spec_sub).issubset(set(self.outlier_set_at))):
            error_spec_tol = 0
            if self.outlier_set_at == self.outlier_set_spec:
                error_spec = 0
        error_general = 1
        if self.outlier_set_at == self.outlier_set_true:
            error_general = 0
        return error_general, error_spec_tol, error_spec

class WRR:
    def __init__(self, instance, mean, std, k, sigma, delta, tol):
        self.instance = instance
        n = len(instance)
        self.n = n
        self.mean = mean
        self.std = std
        self.k = k
        self.threshold_true = self.mean + self.k * self.std
        self.outlier_set_true = []
        self.instance_type = 'bernoulli'
        self.sigma = sigma
        self.delta = delta
        self.tol = tol
        self.t = 0
        self.active_set = []
        self.l_k = 0
        self.rho = 0
        self.wins = np.zeros(n)
        self.pulls = np.ones(n)
        self.rewards = np.zeros(n)
        self.ucbs = np.ones(n)
        self.lcbs = np.zeros(n)
        self.threshold_spec = 0
        self.outlier_set_spec = []
        self.outlier_set_spec_sub = []
        self.outlier_set_spec_sup = []
        # at = anytime
        self.threshold_at = 0
        self.threshold_lub = 0
        self.threshold_ucb = 1
        self.outlier_set_at = []
        self.s_active_len = n
        self.index_pull = 0
        self.threshold_pulls = np.zeros(n)
        # in certain iteration of WRR, pulls on a certain arm need to exceed the threshold pull before pulling other arms

    def compute_ci_hoeffding(self, arm):
        log_term = math.log((np.pi**2 * (self.n+1) * (self.pulls[arm] ** 2)) / (3 * self.delta))
        return math.sqrt(log_term / (2 * self.pulls[arm]))
    # note that they need a (n+1) term rather than a (n) term

    def compute_ci_threshold(self):
        log_term = math.log(((math.pi ** 2) * self.n * (self.t ** 2)) / (3 * self.delta))
        sum_inverse = 0
        for arm in range(self.n):
            sum_inverse += 1 / self.pulls[arm]
        harmonic_mean = self.n / sum_inverse
        return math.sqrt((log_term * self.l_k) / (2 * harmonic_mean))

    def compute_rho(self):
        rho = (((self.n - 1) ** 2) / self.l_k) ** (1 / 3)
        return rho

    def initialization(self):
        g_k = ((1 + (self.k * math.sqrt(self.n - 1))) ** 2) / self.n
        self.l_k = (np.sqrt(g_k) +  np.sqrt(self.k**2/(2* np.log((np.pi**2 * self.n**3)/(6*self.delta)))))**2
        self.rho = self.compute_rho()
        self.threshold_spec = np.mean(self.instance) + self.k * np.std(self.instance)
        for arm in range(self.n):
            if self.instance[arm] > self.threshold_spec - self.tol:
                self.outlier_set_spec_sup.append(arm)
                if self.instance[arm] > self.threshold_spec:
                    self.outlier_set_spec.append(arm)
                    if self.instance[arm] > self.threshold_spec + self.tol:
                        self.outlier_set_spec_sub.append(arm)
            if self.instance[arm] > self.threshold_true:
                self.outlier_set_true.append(arm)
        for arm in range(self.n):
            rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
            self.t += 1
            self.wins[arm] += rwd
            self.pulls[arm] += 1
            self.rewards[arm] = self.wins[arm] / self.pulls[arm]
            beta_tilde = self.compute_ci_hoeffding(arm)
            self.ucbs[arm] = self.rewards[arm] + beta_tilde
            self.lcbs[arm] = self.rewards[arm] - beta_tilde
        beta_tilde_threshold = self.compute_ci_threshold()
        self.threshold_at = np.mean(self.rewards) + self.k * np.std(self.rewards)
        self.ucb_threshold = self.threshold_at + beta_tilde_threshold
        self.lcb_threshold = self.threshold_at - beta_tilde_threshold
        self.outlier_set_at = []
        for arm in range(self.n):
            if self.rewards[arm] > self.threshold_at:
                self.outlier_set_at.append(arm)
            if (self.rewards[arm] > self.threshold_at and self.lcbs[arm] < self.ucb_threshold) or \
                    (self.rewards[arm] <= self.threshold_at and self.ucbs[arm] > self.ucb_threshold):
                self.active_set.append(arm)
        self.s_active_len = len(self.active_set)

    def update_regular(self):
        arm = self.index_pull
        self.threshold_pulls[arm] += self.rho
        rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
        self.t += 1
        self.wins[arm] += rwd
        self.pulls[arm] += 1
        self.rewards[arm] = self.wins[arm] / self.pulls[arm]
        beta_tilde = self.compute_ci_hoeffding(arm)
        self.ucbs[arm] = self.rewards[arm] + beta_tilde
        self.lcbs[arm] = self.rewards[arm] - beta_tilde
        beta_tilde_threshold = self.compute_ci_threshold()
        self.threshold_at = np.mean(self.rewards) + self.k * np.std(self.rewards)
        self.threshold_ucb = self.threshold_at + beta_tilde_threshold
        self.threshold_lcb = self.threshold_at - beta_tilde_threshold
        self.outlier_set_at = []
        for i in range(self.n):
            if self.rewards[i] > self.threshold_at:
                self.outlier_set_at.append(i)
        for arm in self.active_set:
            if (self.rewards[arm] > self.threshold_at and self.threshold_ucb <= self.lcbs[arm]) or \
                    (self.rewards[arm] <= self.threshold_at and self.ucbs[arm] <= self.threshold_lcb):
                self.active_set.remove(arm)
        self.s_active_len = len(self.active_set)
        self.index_pull = (self.index_pull + 1) % self.n

    def update_additional(self, arm):

        rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
        self.t += 1
        self.wins[arm] += rwd
        self.pulls[arm] += 1
        self.rewards[arm] = self.wins[arm] / self.pulls[arm]
        beta_tilde = self.compute_ci_hoeffding(arm)
        self.ucbs[arm] = self.rewards[arm] + beta_tilde
        self.lcbs[arm] = self.rewards[arm] - beta_tilde
        beta_tilde_threshold = self.compute_ci_threshold()
        self.threshold_at = np.mean(self.rewards) + self.k * np.std(self.rewards)
        self.ucb_threshold = self.threshold_at + beta_tilde_threshold
        self.lcb_threshold = self.threshold_at - beta_tilde_threshold
        self.outlier_set_at = []
        for i in range(self.n):
            if self.rewards[i] > self.threshold_at:
                self.outlier_set_at.append(i)
        for arm in self.active_set:
            if (self.rewards[arm] > self.threshold_at and self.threshold_ucb <= self.lcbs[arm]) or \
                    (self.rewards[arm] <= self.threshold_at and self.ucbs[arm] <= self.threshold_lcb):
                self.active_set.remove(arm)
        self.s_active_len = len(self.active_set)

    def compute_error(self):
        error_spec_tol = 1
        error_spec = 1
        if (set(self.outlier_set_at).issubset(set(self.outlier_set_spec_sup))) and \
            (set(self.outlier_set_spec_sub).issubset(set(self.outlier_set_at))):
            error_spec_tol = 0
            if self.outlier_set_at == self.outlier_set_spec:
                error_spec = 0
        error_general = 1
        if self.outlier_set_at == self.outlier_set_true:
            error_general = 0
        return error_general, error_spec_tol, error_spec

    def get_results(self):
        empirical_outlier_set = []
        for arm in range(self.n):
            if self.rewards[arm] > self.threshold:
                empirical_outlier_set.append(arm)
        return empirical_outlier_set, self.t, self.threshold

    def output_outlier_set(self):
        self.threshold = np.mean(self.rewards) + self.k * np.std(self.rewards)
        empirical_outlier_set = []
        for arm in range(self.n):
            if self.rewards[arm] > self.threshold:
                empirical_outlier_set.append(arm)
        return empirical_outlier_set

class RANDOM:
    def __init__(self, instance, n_select, mean, std, k, instance_type, sigma, delta, tol):
        self.n = len(instance)
        n = self.n
        self.instance = instance
        self.n_select = n_select
        self.mean = mean
        self.std = std
        self.k_original = k
        self.threshold_true = self.mean + self.k_original * self.std
        self.outlier_set_true = []
        self.k = 1.4826 * self.k_original
        # k_original denotes the original k
        # while k denotes the adjusted one for MAD
        self.instance_type = instance_type
        self.sigma = sigma
        self.delta = delta
        self.tol = tol
        self.t = 0
        self.wins = np.zeros(n)
        self.pulls = np.zeros(n)
        self.rewards = np.zeros(n)
        self.ucbs = np.ones(n)
        self.lcbs = np.zeros(n)
        self.sample_candidate = list(range(n))

        # s: set; u: upper; m: median; l: lower
        # MAD: median absolute deviation

        self.index_select = []
        self.cluster_boundary_spec = []
        # cluster boundary store boundaries for the selected index
        # everything below are primarily designed for the selected index
        # we will use spec to denote method specific values

        self.s_u_spec = []
        self.s_m_spec = []
        self.s_l_spec = []
        self.median_spec = 0
        # AD = absolute deviation
        self.AD_spec = np.zeros(n)
        self.s_MAD_spec = []
        self.MAD_spec = 0
        self.threshold_spec = 0
        self.outlier_set_spec = []
        self.outlier_set_spec_sub = []
        self.outlier_set_spec_sup = []
        # at = anytime
        # anytime here refers to anytime decision of the set
        self.s_u_at = []
        self.s_m_at = []
        self.s_l_at = []
        self.median_at = 0
        self.AD_at = np.zeros(n)
        self.s_MAD_at = []
        self.MAD_at = 0
        self.threshold_at = 0
        # s_median_ucb store arms contribute to the ucb of median
        self.s_median_ucb = []
        self.median_ucb = 1
        self.s_median_lcb = []
        self.median_lcb = 0
        self.AD_ucbs = np.ones(n)
        self.AD_lcbs = np.ones(n)
        # s_MAD_ucb store arms contribute to the ucb of MAD
        self.s_MAD_ucb = []
        self.MAD_ucb = 1
        self.s_MAD_lcb = []
        self.MAD_lcb = 0
        self.threshold_lcb = 0
        self.threshold_ucb = 1
        self.s_active = []
        self.s_active_len = n
        # store active arms


    def compute_ci_hoeffding(self, arm):
        beta = math.log((np.pi**2 * (self.n) * (self.pulls[arm] ** 2)) / (3 * self.delta))
        return math.sqrt(beta / (2 * self.pulls[arm]))

    def compute_ci_subgaussian(self, arm):
        log_term = math.log((np.pi**2 * (self.n) * (self.pulls[arm] ** 2)) / (3 * self.delta))
        return self.sigma * math.sqrt(2 * log_term / self.pulls[arm])

    def update_internal(self):
        [start, end] = self.cluster_boundary_spec
        ranking_lcbs = np.argsort(self.lcbs[self.index_select])
        self.s_median_lcb = self.index_select[ranking_lcbs[start: end]]
        self.median_lcb = sum(self.lcbs[i] for i in self.s_median_lcb) / len(self.s_median_lcb)
        ranking_ucbs = np.argsort(self.ucbs[self.index_select])
        self.s_median_ucb = self.index_select[ranking_ucbs[start: end]]
        self.median_ucb = sum(self.ucbs[i] for i in self.s_median_ucb) / len(self.s_median_ucb)
        for i in self.index_select:
            self.AD_ucbs[i] = max(self.ucbs[i] - self.median_lcb, self.median_ucb - self.lcbs[i])
            self.AD_lcbs[i] = max(self.lcbs[i] - self.median_ucb, self.median_lcb - self.ucbs[i])
            # we define AD_lcb in the way above to provide better estimations of \widehat{AD} at the beginning stage
            # if self.ucbs[i] >= self.median_ucb:
            #     if self.median_ucb <= self.lcbs[i]:
            #         self.AD_lcbs[i] = self.lcbs[i] - self.median_ucb
            #     else:
            #         self.AD_lcbs[i] = 0
            # else:
            #     if self.ucbs[i] <= self.median_lcb:
            #         self.AD_lcbs[i] = self.median_lcb - self.ucbs[i]
            #     else:
            #         self.AD_lcbs[i] = 0
            if self.AD_ucbs[i] < self.AD_lcbs[i]:
                print('something wrong when computing the absolute deviation')
        ranking_AD_lcbs = np.argsort(self.AD_lcbs[self.index_select])
        self.s_MAD_lcb = self.index_select[ranking_AD_lcbs[start: end]]
        self.MAD_lcb = sum(self.AD_lcbs[i] for i in self.s_MAD_lcb) / len(self.s_MAD_lcb)
        ranking_AD_ucbs = np.argsort(self.AD_ucbs[self.index_select])
        self.s_MAD_ucb = self.index_select[ranking_AD_ucbs[start: end]]
        self.MAD_ucb = sum(self.AD_ucbs[i] for i in self.s_MAD_ucb) / len(self.s_MAD_ucb)
        self.threshold_lcb = self.median_lcb + self.k * self.MAD_lcb
        self.threshold_ucb = self.median_ucb + self.k * self.MAD_ucb
        self.threshold_at = (self.threshold_lcb + self.threshold_ucb)/2
        self.s_active = list(range(self.n))
        for i in range(self.n):
            if self.ucbs[i] < self.threshold_lcb or self.lcbs[i] > self.threshold_ucb:
                self.s_active.remove(i)
        self.s_active_len = len(self.s_active)


    def update(self):
        arm = np.random.choice(self.sample_candidate)
        rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
        self.t += 1
        self.wins[arm] += rwd
        self.pulls[arm] += 1
        self.rewards[arm] = self.wins[arm] / self.pulls[arm]
        if self.instance_type == 'bernoulli':
            beta_tilde = self.compute_ci_hoeffding(arm)
        else:
            beta_tilde = self.compute_ci_subgaussian(arm)

        self.ucbs[arm] = self.rewards[arm] + beta_tilde
        self.lcbs[arm] = self.rewards[arm] - beta_tilde
        if arm in self.index_select:
            self.update_internal()


    def initialization(self):

        self.index_select = np.random.choice(self.n, self.n_select, replace=False)
        n_select = self.n_select
        if n_select % 2 == 1:
            start = int((n_select - 1) / 2)
            end = int((n_select + 1) / 2)
        else:
            start = int((n_select - 2) / 2)
            end = int((n_select + 2) / 2)
        self.cluster_boundary_spec = [start, end]
        ranking = np.argsort(self.instance[self.index_select])
        self.s_l_spec = self.index_select[ranking[:start]]
        self.s_m_spec = self.index_select[ranking[start:end]]
        self.s_u_spec = self.index_select[ranking[end:]]
        self.median_spec = sum(self.instance[i] for i in self.s_m_spec) / len(self.s_m_spec)
        for i in range(self.n):
            self.AD_spec[i] = abs(self.instance[i] - self.median_spec)
        ranking_AD = np.argsort(self.AD_spec[self.index_select])
        self.s_MAD_spec = self.index_select[ranking_AD[start:end]]
        self.MAD_spec = sum(self.AD_spec[i] for i in self.s_MAD_spec) / len(self.s_MAD_spec)
        self.threshold_spec = self.median_spec + self.k * self.MAD_spec
        for arm in range(self.n):
            if self.instance[arm] > self.threshold_spec - self.tol:
                self.outlier_set_spec_sup.append(arm)
                if self.instance[arm] > self.threshold_spec:
                    self.outlier_set_spec.append(arm)
                    if self.instance[arm] > self.threshold_spec + self.tol:
                        self.outlier_set_spec_sub.append(arm)
            if self.instance[arm] > self.threshold_true:
                self.outlier_set_true.append(arm)
        # pull each arm once
        for arm in range(self.n):
            rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
            self.t += 1
            self.wins[arm] += rwd
            self.pulls[arm] += 1
            self.rewards[arm] = self.wins[arm] / self.pulls[arm]
            if self.instance_type == 'bernoulli':
                beta_tilde = self.compute_ci_hoeffding(arm)
            else:
                beta_tilde = self.compute_ci_subgaussian(arm)
            self.ucbs[arm] = self.rewards[arm] + beta_tilde
            self.lcbs[arm] = self.rewards[arm] - beta_tilde
        self.update_internal()

    def output_outlier_set(self):
        # output empirical outlier set
        outlier_set_empirical = []
        for arm in range(self.n):
            if self.rewards[arm] > self.threshold_at:
                outlier_set_empirical.append(arm)
        return outlier_set_empirical

    def compute_error(self):
        outlier_set_at = self.output_outlier_set()
        error_spec_tol = 1
        error_spec = 1
        if (set(outlier_set_at).issubset(set(self.outlier_set_spec_sup))) and \
            (set(self.outlier_set_spec_sub).issubset(set(outlier_set_at))):
            error_spec_tol = 0
            if outlier_set_at == self.outlier_set_spec:
                error_spec = 0
        error_general = 1
        if outlier_set_at == self.outlier_set_true:
            error_general = 0
        return error_general, error_spec_tol, error_spec

class ROAI:
    def __init__(self, instance, n_select, mean, std, k, instance_type, sigma, delta, tol):
        self.n = len(instance)
        n = self.n
        self.instance = instance
        self.n_select = n_select
        self.mean = mean
        self.std = std
        self.k_original = k
        self.threshold_true = self.mean + self.k_original * self.std
        self.outlier_set_true = []
        self.k = 1.4826 * k
        # k_original denotes the original k value
        # while k is adjusted for MAD
        self.instance_type = instance_type
        self.sigma = sigma
        self.delta = delta
        self.tol = tol
        self.t = 0
        self.wins = np.zeros(n)
        self.pulls = np.zeros(n)
        self.rewards = np.zeros(n)
        self.ucbs = np.ones(n)
        self.lcbs = np.zeros(n)
        # s: set; u: upper; m: median; l: lower; all in terms of median value
        # MAD: median absolute deviation
        self.index_select = []
        self.cluster_boundary_spec = []
        # cluster boundary store boundaries for the selected index
        # everything below are primarily designed for the selected index
        # we will use spec to denote the method-specific
        self.s_u_spec = []
        self.s_m_spec = []
        self.s_l_spec = []
        self.median_spec = 0
        # AD = absolute deviation
        self.AD_spec = np.zeros(n)
        self.s_MAD_spec = []
        self.MAD_spec = 0
        self.threshold_spec = 0
        self.outlier_set_spec = []
        self.outlier_set_spec_sub = []
        self.outlier_set_spec_sup = []
        # calculated based on the specific way of selecting outlier threshold
        # at = anytime
        # anytime here refers to anytime decision of the set
        self.s_u_at = []
        self.s_m_at = []
        self.s_l_at = []
        self.median_at = 0
        self.AD_at = np.zeros(n)
        self.s_MAD_at = []
        self.MAD_at = 0
        self.threshold_at = 0
        # sample_candidate is the set for arms to be sampled, which should be the union of three components
        self.sample_candidate = []
        self.sample_candidate_threshold = []
        self.sample_candidate_arms = []
        # median lcb = median(lcbs), same for median_ucb
        self.s_median_lcb = []
        self.median_lcb = 0
        self.s_median_ucb = []
        self.median_ucb = 1
        # AD_ucbs = upper bound of absolute deviation, same for AD_lcbs
        self.AD_ucbs = np.ones(n)
        self.AD_lcbs = np.ones(n)
        # s_MAD_ucb contains arms that contribute to the ucb of MAD
        self.s_MAD_ucb = []
        self.MAD_ucb = 1
        self.s_MAD_lcb = []
        self.MAD_lcb = 0
        self.threshold_lcb = 0
        self.threshold_ucb = 1
        # below are for lucb algorithm
        self.s_outlier_at = []
        self.s_not_outlier_at = []
        # upper/lower set of arms in terms of AD at anytime; same as self.s_MAD_at
        self.s_uAD_at = []
        self.s_lAD_at = []
        self.s_active = []
        # arms in active set are those haven't been determined
        self.s_active_len = n



    def compute_ci_hoeffding(self, arm):
        log_term = math.log((np.pi**2 * (self.n) * (self.pulls[arm] ** 2)) / (3 * self.delta))
        return math.sqrt(log_term / (2 * self.pulls[arm]))

    def compute_ci_subgaussian(self, arm):
        log_term = math.log((np.pi**2 * (self.n) * (self.pulls[arm] ** 2)) / (3 * self.delta))
        return self.sigma * math.sqrt(2 * log_term / self.pulls[arm])

    # update threshold and sample candidate
    # _elimi = elimination-styled updating in how to select sample candidate
    def update_internal_elimi(self):
        [start, end] = self.cluster_boundary_spec
        ranking_lcbs = np.argsort(self.lcbs[self.index_select])
        self.s_median_lcb = self.index_select[ranking_lcbs[start: end]]
        self.median_lcb = sum(self.lcbs[i] for i in self.s_median_lcb) / len(self.s_median_lcb)
        ranking_ucbs = np.argsort(self.ucbs[self.index_select])
        self.s_median_ucb = self.index_select[ranking_ucbs[start: end]]
        self.median_ucb = sum(self.ucbs[i] for i in self.s_median_ucb) / len(self.s_median_ucb)
        for i in self.index_select:
            self.AD_ucbs[i] = max(self.ucbs[i] - self.median_lcb, self.median_ucb - self.lcbs[i])
            self.AD_lcbs[i] = max(self.lcbs[i] - self.median_ucb, self.median_lcb - self.ucbs[i])
            # we define AD_lcb in the way above to provide better estimations of \widehat{AD} at the beginning stage
            # if self.ucbs[i] >= self.median_ucb:
            #     if self.median_ucb <= self.lcbs[i]:
            #         self.AD_lcbs[i] = self.lcbs[i] - self.median_ucb
            #     else:
            #         self.AD_lcbs[i] = 0
            # else:
            #     if self.ucbs[i] <= self.median_lcb:
            #         self.AD_lcbs[i] = self.median_lcb - self.ucbs[i]
            #     else:
            #         self.AD_lcbs[i] = 0
            if self.AD_ucbs[i] < self.AD_lcbs[i]:
                print('something wrong when computing the absolute deviation')
        ranking_AD_lcbs = np.argsort(self.AD_lcbs[self.index_select])
        self.s_MAD_lcb = self.index_select[ranking_AD_lcbs[start: end]]
        self.MAD_lcb = sum(self.AD_lcbs[i] for i in self.s_MAD_lcb) / len(self.s_MAD_lcb)
        ranking_AD_ucbs = np.argsort(self.AD_ucbs[self.index_select])
        self.s_MAD_ucb = self.index_select[ranking_AD_ucbs[start: end]]
        self.MAD_ucb = sum(self.AD_ucbs[i] for i in self.s_MAD_ucb) / len(self.s_MAD_ucb)
        self.threshold_lcb = self.median_lcb + self.k * self.MAD_lcb
        self.threshold_ucb = self.median_ucb + self.k * self.MAD_ucb
        self.threshold_at = (self.threshold_lcb + self.threshold_ucb)/2

        # arms whose confidence interval intersects with the ci of threshold should be sample candidates

        self.s_active = list(range(self.n))
        for i in range(self.n):
            if self.ucbs[i] < self.threshold_lcb or self.lcbs[i] > self.threshold_ucb:
                self.s_active.remove(i)
        self.s_active_len = len(self.s_active)

        self.sample_candidate = list(range(self.n))
        for i in range(self.n):
            if self.ucbs[i] < self.threshold_lcb or self.lcbs[i] > self.threshold_ucb:
                self.sample_candidate.remove(i)

        # things below are specified for the elimination style

        for i in self.index_select:
            if (self.ucbs[i] >= self.median_ucb and self.lcbs[i] <= self.median_ucb) \
                    or (self.ucbs[i] < self.median_ucb and self.ucbs[i] >= self.median_lcb):
                self.sample_candidate.append(i)

            if (self.AD_ucbs[i] >= self.MAD_ucb and self.AD_lcbs[i] <= self.MAD_ucb) \
                    or (self.AD_ucbs[i] < self.MAD_ucb and self.AD_ucbs[i] >= self.MAD_lcb):
                self.sample_candidate.append(i)

        self.sample_candidate = list(set(self.sample_candidate))
        self.sample_candidate.sort()

    def initialization_elimi(self):
        self.index_select = np.random.choice(self.n, self.n_select, replace=False)
        # compute true set of outliers
        n = self.n_select
        if n % 2 == 1:
            start = int((n - 1) / 2)
            end = int((n + 1) / 2)
        else:
            start = int((n - 2) / 2)
            end = int((n + 2) / 2)
        self.cluster_boundary_spec = [start, end]
        ranking = np.argsort(self.instance[self.index_select])
        self.s_l_spec = self.index_select[ranking[:start]]
        self.s_m_spec = self.index_select[ranking[start:end]]
        self.s_u_spec = self.index_select[ranking[end:]]
        self.median_spec = sum(self.instance[i] for i in self.s_m_spec) / len(self.s_m_spec)
        for i in range(self.n):
            self.AD_spec[i] = abs(self.instance[i] - self.median_spec)
        ranking_AD = np.argsort(self.AD_spec[self.index_select])
        self.s_MAD_spec = self.index_select[ranking_AD[start:end]]
        self.MAD_spec = sum(self.AD_spec[i] for i in self.s_MAD_spec) / len(self.s_MAD_spec)
        self.threshold_spec = self.median_spec + self.k * self.MAD_spec
        for arm in range(self.n):
            if self.instance[arm] > self.threshold_spec - self.tol:
                self.outlier_set_spec_sup.append(arm)
                if self.instance[arm] > self.threshold_spec:
                    self.outlier_set_spec.append(arm)
                    if self.instance[arm] > self.threshold_spec + self.tol:
                        self.outlier_set_spec_sub.append(arm)
            if self.instance[arm] > self.threshold_true:
                self.outlier_set_true.append(arm)
        # pull each arm once
        for arm in range(self.n):
            rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
            self.t += 1
            self.wins[arm] += rwd
            self.pulls[arm] += 1
            self.rewards[arm] = self.wins[arm] / self.pulls[arm]
            if self.instance_type == 'bernoulli':
                beta_tilde = self.compute_ci_hoeffding(arm)
            else:
                beta_tilde = self.compute_ci_subgaussian(arm)
            self.ucbs[arm] = self.rewards[arm] + beta_tilde
            self.lcbs[arm] = self.rewards[arm] - beta_tilde

        self.update_internal_elimi()


    def update_elimi(self):
        self.t += 1
        if len(self.sample_candidate) > 0:
            arm = np.random.choice(self.sample_candidate)
            rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
            self.pulls[arm] += 1
            self.wins[arm] += rwd
            self.rewards[arm] = self.wins[arm] / self.pulls[arm]
            if self.instance_type == 'bernoulli':
                beta_tilde = self.compute_ci_hoeffding(arm)
            else:
                beta_tilde = self.compute_ci_subgaussian(arm)
            self.ucbs[arm] = self.rewards[arm] + beta_tilde
            self.lcbs[arm] = self.rewards[arm] - beta_tilde
            self.update_internal_elimi()

# lucb styled algorithm
    def update_internal_lucb(self):
        [start, end] = self.cluster_boundary_spec
        ranking_lcbs = np.argsort(self.lcbs[self.index_select])
        self.s_median_lcb = self.index_select[ranking_lcbs[start: end]]
        self.median_lcb = sum(self.lcbs[i] for i in self.s_median_lcb) / len(self.s_median_lcb)
        ranking_ucbs = np.argsort(self.ucbs[self.index_select])
        self.s_median_ucb = self.index_select[ranking_ucbs[start: end]]
        self.median_ucb = sum(self.ucbs[i] for i in self.s_median_ucb) / len(self.s_median_ucb)
        for i in self.index_select:
            self.AD_ucbs[i] = max(self.ucbs[i] - self.median_lcb, self.median_ucb - self.lcbs[i])
            self.AD_lcbs[i] = max(self.lcbs[i] - self.median_ucb, self.median_lcb - self.ucbs[i])
            # we define AD_lcb in the way above to provide better estimations of \widehat{AD} at the beginning stage
            # if self.ucbs[i] >= self.median_ucb:
            #     if self.median_ucb <= self.lcbs[i]:
            #         self.AD_lcbs[i] = self.lcbs[i] - self.median_ucb
            #     else:
            #         self.AD_lcbs[i] = 0
            # else:
            #     if self.ucbs[i] <= self.median_lcb:
            #         self.AD_lcbs[i] = self.median_lcb - self.ucbs[i]
            #     else:
            #         self.AD_lcbs[i] = 0
            if self.AD_ucbs[i] < self.AD_lcbs[i]:
                print('something wrong when computing the absolute deviation')
        ranking_AD_lcbs = np.argsort(self.AD_lcbs[self.index_select])
        self.s_MAD_lcb = self.index_select[ranking_AD_lcbs[start: end]]
        self.MAD_lcb = sum(self.AD_lcbs[i] for i in self.s_MAD_lcb) / len(self.s_MAD_lcb)
        ranking_AD_ucbs = np.argsort(self.AD_ucbs[self.index_select])
        self.s_MAD_ucb = self.index_select[ranking_AD_ucbs[start: end]]
        self.MAD_ucb = sum(self.AD_ucbs[i] for i in self.s_MAD_ucb) / len(self.s_MAD_ucb)
        self.threshold_lcb = self.median_lcb + self.k * self.MAD_lcb
        self.threshold_ucb = self.median_ucb + self.k * self.MAD_ucb
        self.threshold_at = (self.threshold_lcb + self.threshold_ucb)/2
        self.s_active = list(range(self.n))
        for i in range(self.n):
            if self.ucbs[i] < self.threshold_lcb or self.lcbs[i] > self.threshold_ucb:
                self.s_active.remove(i)
        self.s_active_len = len(self.s_active)
        ranking_means = np.argsort(self.rewards[self.index_select])
        self.s_l_at = self.index_select[ranking_means[:start]]
        self.s_m_at = self.index_select[ranking_means[start:end]]
        self.s_u_at = self.index_select[ranking_means[end:]]
        self.median_at = sum(self.rewards[i] for i in self.s_m_at) / len(self.s_m_at)
        for i in self.index_select:
            self.AD_at[i] = (self.AD_lcbs[i] + self.AD_ucbs[i])/2
            # self.AD_at[i] = abs(self.rewards[i] - self.median_at)
            # one can also calculate \hat{AD} in the commented way and it produce slightly better results in the beginning period
        # s_lAD_at denote the set of arms associated with low value of AD
        ranking_AD = np.argsort(self.AD_at[self.index_select])
        self.s_lAD_at = self.index_select[ranking_AD[:start]]
        self.s_MAD_at = self.index_select[ranking_AD[start:end]]
        self.s_uAD_at = self.index_select[ranking_AD[end:]]
        self.MAD_at = sum(self.AD_at[i] for i in self.s_MAD_at) / len(self.s_MAD_at)
        # self.threshold_at = self.median_at + self.k * self.MAD_at
        self.s_outlier_at = []
        self.s_not_outlier_at = []
        for arm in range(self.n):
            if self.rewards[arm] >= self.threshold_at:
                self.s_outlier_at.append(arm)
            else:
                self.s_not_outlier_at.append(arm)

        # things below are specified for the lucb style
        # arms whose confidence interval intersects with the c.i. of threshold should be sample candidates
        self.sample_candidate = []
        self.sample_candidate_arms = []
        self.sample_candidate_threshold = []

        s_outlier_at = self.s_outlier_at
        s_candidate = [(x, self.lcbs[x]) for x in s_outlier_at]
        if len(s_candidate) > 0:
            candidate_value = min(s_candidate, key=lambda x: x[1])[1]
            s_candidate_index = [x for x,y in s_candidate if y == candidate_value]
            candidate = np.random.choice(s_candidate_index)
            self.sample_candidate.append(candidate)
            self.sample_candidate_arms.append(candidate)

        s_not_outlier_at = self.s_not_outlier_at
        s_candidate = [(x, self.ucbs[x]) for x in s_not_outlier_at]
        if len(s_candidate) > 0:
            candidate_value = max(s_candidate, key=lambda x: x[1])[1]
            s_candidate_index = [x for x,y in s_candidate if y == candidate_value]
            candidate = np.random.choice(s_candidate_index)
            self.sample_candidate.append(candidate)
            self.sample_candidate_arms.append(candidate)

        s_l_at = self.s_l_at
        s_candidate = [(x, self.ucbs[x]) for x in s_l_at]
        if len(s_candidate) > 0:
            candidate_value = max(s_candidate, key=lambda x: x[1])[1]
            s_candidate_index = [x for x,y in s_candidate if y == candidate_value]
            candidate = np.random.choice(s_candidate_index)
            self.sample_candidate.append(candidate)
            self.sample_candidate_threshold.append(candidate)

        s_u_at = self.s_u_at
        s_candidate = [(x, self.lcbs[x]) for x in s_u_at]
        if len(s_candidate) > 0:
            candidate_value = min(s_candidate, key=lambda x: x[1])[1]
            s_candidate_index = [x for x,y in s_candidate if y == candidate_value]
            candidate = np.random.choice(s_candidate_index)
            self.sample_candidate.append(candidate)
            self.sample_candidate_threshold.append(candidate)

        s_lAD_at = self.s_lAD_at
        s_candidate = [(x, self.AD_ucbs[x]) for x in s_lAD_at]
        if len(s_candidate) > 0:
            candidate_value = max(s_candidate, key=lambda x: x[1])[1]
            s_candidate_index = [x for x,y in s_candidate if y == candidate_value]
            candidate = np.random.choice(s_candidate_index)
            self.sample_candidate.append(candidate)
            self.sample_candidate_threshold.append(candidate)

        s_uAD_at = self.s_uAD_at
        s_candidate = [(x, self.AD_lcbs[x]) for x in s_uAD_at]
        if len(s_candidate) > 0:
            candidate_value = min(s_candidate, key=lambda x: x[1])[1]
            s_candidate_index = [x for x,y in s_candidate if y == candidate_value]
            candidate = np.random.choice(s_candidate_index)
            self.sample_candidate.append(candidate)
            self.sample_candidate_threshold.append(candidate)


        s_lm_at = set(list(self.s_l_at) + list(self.s_m_at))
        s_candidate = [(x, self.ucbs[x]) for x in s_lm_at]
        if len(s_candidate) > 0:
            candidate_value = max(s_candidate, key=lambda x: x[1])[1]
            s_candidate_index = [x for x,y in s_candidate if y == candidate_value]
            candidate = np.random.choice(s_candidate_index)
            self.sample_candidate.append(candidate)
            self.sample_candidate_threshold.append(candidate)

        s_um_at = set(list(self.s_u_at) + list(self.s_m_at))
        s_candidate = [(x, self.lcbs[x]) for x in s_um_at]
        if len(s_candidate) > 0:
            candidate_value = min(s_candidate, key=lambda x: x[1])[1]
            s_candidate_index = [x for x,y in s_candidate if y == candidate_value]
            candidate = np.random.choice(s_candidate_index)
            self.sample_candidate.append(candidate)
            self.sample_candidate_threshold.append(candidate)

        s_lmAD_at = set(list(self.s_lAD_at) + list(self.s_MAD_at))
        s_candidate = [(x, self.AD_ucbs[x]) for x in s_lmAD_at]
        if len(s_candidate) > 0:
            candidate_value = max(s_candidate, key=lambda x: x[1])[1]
            s_candidate_index = [x for x,y in s_candidate if y == candidate_value]
            candidate = np.random.choice(s_candidate_index)
            self.sample_candidate.append(candidate)
            self.sample_candidate_threshold.append(candidate)

        s_umAD_at = set(list(self.s_uAD_at) + list(self.s_MAD_at))
        s_candidate = [(x, self.AD_lcbs[x]) for x in s_umAD_at]
        if len(s_candidate) > 0:
            candidate_value = min(s_candidate, key=lambda x: x[1])[1]
            s_candidate_index = [x for x,y in s_candidate if y == candidate_value]
            candidate = np.random.choice(s_candidate_index)
            self.sample_candidate.append(candidate)
            self.sample_candidate_threshold.append(candidate)

        #self.sample_candidate = list(set(self.sample_candidate))
        # since we only pull one arm at each time in experiment, we will allow repeated arms in sample candidate
        # actually that's more fair to increase the prob to select that arm

    def initialization_lucb(self):
        self.index_select = np.random.choice(self.n, self.n_select, replace=False)
        # compute true set of outliers
        n = self.n_select
        if n % 2 == 1:
            start = int((n - 1) / 2)
            end = int((n + 1) / 2)
        else:
            start = int((n - 2) / 2)
            end = int((n + 2) / 2)
        self.cluster_boundary_spec = [start, end]
        ranking = np.argsort(self.instance[self.index_select])
        self.s_l_spec = self.index_select[ranking[:start]]
        self.s_m_spec = self.index_select[ranking[start:end]]
        self.s_u_spec = self.index_select[ranking[end:]]
        self.median_spec = sum(self.instance[i] for i in self.s_m_spec) / len(self.s_m_spec)
        for i in range(self.n):
            self.AD_spec[i] = abs(self.instance[i] - self.median_spec)
        ranking_AD = np.argsort(self.AD_spec[self.index_select])
        self.s_MAD_spec = self.index_select[ranking_AD[start:end]]
        self.MAD_spec = sum(self.AD_spec[i] for i in self.s_MAD_spec) / len(self.s_MAD_spec)
        self.threshold_spec = self.median_spec + self.k * self.MAD_spec
        for arm in range(self.n):
            if self.instance[arm] > self.threshold_true:
                self.outlier_set_true.append(arm)
            if self.instance[arm] > self.threshold_spec - self.tol:
                self.outlier_set_spec_sup.append(arm)
                if self.instance[arm] > self.threshold_spec:
                    self.outlier_set_spec.append(arm)
                    if self.instance[arm] > self.threshold_spec + self.tol:
                        self.outlier_set_spec_sub.append(arm)
        # pull each arm once
        for arm in range(self.n):
            rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
            self.t += 1
            self.wins[arm] += rwd
            self.pulls[arm] += 1
            self.rewards[arm] = self.wins[arm] / self.pulls[arm]
            if self.instance_type == 'bernoulli':
                beta_tilde = self.compute_ci_hoeffding(arm)
            else:
                beta_tilde = self.compute_ci_subgaussian(arm)
            self.ucbs[arm] = self.rewards[arm] + beta_tilde
            self.lcbs[arm] = self.rewards[arm] - beta_tilde

        self.update_internal_lucb()


    def update_lucb(self):
        self.t += 1
        if len(self.sample_candidate) > 0:

            # if len(self.sample_candidate_arms) > 0 and len(self.sample_candidate_threshold) > 0:
            #     dice = np.random.random()
            #     if dice > 0.5:
            #         arm = np.random.choice(self.sample_candidate_threshold)
            #     else:
            #         arm = np.random.choice(self.sample_candidate_arms)
            # else:
            #     arm = np.random.choice(self.sample_candidate)
            arm = np.random.choice(self.sample_candidate)
            # randomly pull an arm from sample_candidate
            rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
            self.pulls[arm] += 1
            self.wins[arm] += rwd
            self.rewards[arm] = self.wins[arm] / self.pulls[arm]
            if self.instance_type == 'bernoulli':
                beta_tilde = self.compute_ci_hoeffding(arm)
            else:
                beta_tilde = self.compute_ci_subgaussian(arm)
            self.ucbs[arm] = self.rewards[arm] + beta_tilde
            self.lcbs[arm] = self.rewards[arm] - beta_tilde
            self.update_internal_lucb()


    def output_outlier_set(self):
        outlier_set_empirical = []
        for arm in range(self.n):
            if self.rewards[arm] > self.threshold_at:
                outlier_set_empirical.append(arm)
        return outlier_set_empirical

    def compute_error(self):
        outlier_set_at = self.output_outlier_set()
        error_spec_tol = 1
        error_spec = 1
        if (set(outlier_set_at).issubset(set(self.outlier_set_spec_sup))) and \
            (set(self.outlier_set_spec_sub).issubset(set(outlier_set_at))):
            error_spec_tol = 0
            if outlier_set_at == self.outlier_set_spec:
                error_spec = 0
        error_general = 1
        if outlier_set_at == self.outlier_set_true:
            error_general = 0


        return error_general, error_spec_tol, error_spec
