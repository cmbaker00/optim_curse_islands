import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import os
from functools import lru_cache


class IslandInvasives:
    def __init__(self, num_islands, budget=200., cost_average=50., cost_variance=40.,
                 value_variance=.1, estimation_variance=.05):
        self.num_islands = num_islands
        self.island_list = list(range(self.num_islands))

        self.cost_average = cost_average  # average cost to remove species
        self.cost_variance = cost_variance  # variance in cost
        self.cost_distribution = "LogNormal"  # distribution to use (Uniform or LogNormal)

        self.value_average = 1  # value of removing species
        self.value_variance = value_variance  # variance in values
        self.value_distribution = "LogNormal"  # distribution to use (Uniform or LogNormal)

        self.estimation_variance = estimation_variance  # variance in estimating cost & value

        self.budget = budget  # total project_budget

        self.islands = None  # true values about islands
        self.estimates = None   # estimated values
        self.island_rankings = None  # Ranking of islands worst to best.

        # placeholders for the list of choices
        self.random_choice = None
        self.costben_choice = None
        self.optimal_choice = None
        self.cheap_choice = None
        self.good_choice = None

    def run_prioritisation(self):  # run the full analysis for a set of islands
        self.generate_islands()
        self.generate_estimates()
        self.choose_random()
        self.choose_costben()
        self.choose_optimal()
        self.choose_cheap()
        self.choose_good()

    def generate_islands(self):  # generate island parameters
        if self.islands is None:
            island_costs = self.generate_costs()
            island_values = self.generate_values()
            # store the costs and values in an array
            self.islands = np.array([island_costs, island_values, island_values/island_costs])
        return

    # The following four functions return the expected/true value/cost of an input list of islands
    def expected_value(self, choice):
        return np.sum(self.estimates[1, choice])

    def expected_cost(self, choice):
        return np.sum(self.estimates[0, choice])

    def true_value(self, choice):
        return np.sum(self.islands[1, choice])

    def true_cost(self, choice):
        return np.sum(self.islands[0, choice])

    def generate_estimates(self):  # generate and store estimates of cost and value
        if self.estimates is None:
            if self.islands is None:
                raise RuntimeError("generate_islands must be called before generate estimates")

            estimates = self.islands*self.beta_estimate_draws(self.islands.shape)
            if (estimates <= 0).any():
                raise ValueError('All value/cost estimates must be positive')
            estimates[2, :] = estimates[1, :]/estimates[0, :]  # store a 3rd row, the benefit/cost of each island
            self.estimates = estimates
        return

    def beta_estimate_draws(self, num):
        # generate estimates with a beta distribution with parameters Beta(a,a)
        # Variance of beta distribution is 1/(4+8a), so a = (1-4var)/(8var). Variance must be less than 0.25
        # However, the beta distribution is 'doubled' to have support [0,2], so that also doubles the variances
        # Hence, variance must be halved at this calculate -> input variance can be up to 0.5
        a = self.beta_parameters(self.estimation_variance)
        return 2*np.random.beta(a, a, num)

    @staticmethod
    @lru_cache()
    def beta_parameters(var):
        if var*2 >= 1:
            raise ValueError("Variance is too large for the beta distribution")
        return (1-2*var)/(4*var)

    def generate_values(self):
        if self.value_distribution is 'Uniform':
            distribution = self.generate_uniform
        elif self.value_distribution is 'LogNormal':
            distribution = self.generate_lognormal
        else:
            raise Exception('Unknown option for value distribution: {}'.format(self.value_distribution))
        return distribution(self.num_islands, self.value_average, self.value_variance)

    def generate_costs(self):
        if self.cost_distribution is 'Uniform':
            distribution = self.generate_uniform
        elif self.cost_distribution is 'LogNormal':
            distribution = self.generate_lognormal
        else:
            raise Exception('Unknown option for value distribution: {}'.format(self.value_distribution))
        return distribution(self.num_islands, self.cost_average, self.cost_variance)

    @staticmethod
    def generate_uniform(num, mean, variance):
        a = mean - np.sqrt(12 * variance)/2
        b = mean + np.sqrt(12 * variance)/2
        return np.random.uniform(a, b, num)

    def generate_lognormal(self, num, mean, variance):
        mu, sigma = self.lognormal_parameters(mean, variance)
        return np.exp(np.random.normal(mu, sigma, num))

    @staticmethod
    @lru_cache()
    def lognormal_parameters(mean, variance):
        # I couldn't find an equation for mu and sigma as a function of
        # mean and variance for the lognormal distribution
        # this uses fixed point iteration
        mu = np.log(mean)  # initial guess of mu
        sigma = None
        flag = 0
        while flag == 0:
            sigma = np.sqrt(np.log(0.5 * (1 + np.sqrt(1 + 4 * variance * np.exp(-2 * mu)))))  # true sigma, given  mu
            prev_mu = copy.copy(mu)
            mu = np.log(mean) - (sigma ** 2) / 2  # update mu.
            if np.abs(prev_mu - mu) < .001:
                flag = 1
            if np.abs(mu) > 1e6:
                raise Exception
        return mu, sigma

    def choose_random(self):  # choose a random set of islands within project_budget
        if self.random_choice is None:
            flag = 0
            islands = copy.copy(self.island_list)  # create a local island list
            random_choice = []  # empty list to be populated with the set of islands
            while flag == 0:
                # add a random island to the choice list and remove it from the local island list
                random_choice.append(islands.pop(islands.index(np.random.choice(islands))))

                current_costs = self.estimates[0, random_choice]
                current_total_cost = np.sum(current_costs)

                if current_total_cost > self.budget:  # check is overbudget
                    random_choice.pop(-1)  # remove the last choice that went overbudget
                    flag = 1
                if len(islands) == 0:
                    flag = 1  # exit if all islands are chosen

            self.random_choice = random_choice
            return random_choice

        return self.random_choice

    def choose_costben(self):  # choose islands by cost vs benefit
        if self.costben_choice is None:
            islands = copy.copy(self.island_list)  # local island list
            islands_bencost = copy.copy(self.estimates[2, :])  # local list of benefit/cost for ech island
            islands_bencost_array = np.array([islands, islands_bencost])  # join them together
            # sort by benefit/cost
            islands_bencost_array = islands_bencost_array[:, islands_bencost_array[1, :].argsort()]

            flag = 0
            # return a list of islands ranked from worst to best
            islands_ranked_worst_to_best = [int(val) for val in islands_bencost_array[0, :]]
            self.island_rankings = copy.copy(islands_ranked_worst_to_best)  # save the list for use elsewhere
            costben_choice = []
            # as per random choice, but always adding the next best island until the project_budget is exhausted
            while flag == 0:
                costben_choice.append(islands_ranked_worst_to_best.pop(-1))
                current_costs = self.estimates[0, costben_choice]
                current_total_cost = np.sum(current_costs)
                if current_total_cost > self.budget:
                    costben_choice.pop(-1)
                    flag = 1
                if len(islands_ranked_worst_to_best) == 0:
                    flag = 1
            self.costben_choice = costben_choice
            return costben_choice
        return self.costben_choice

    def choose_optimal(self):
        if self.optimal_choice is None:
            costben_choice = self.choose_costben()  # start with the costben solution and explore 'nearby' options
            current_best_choice = costben_choice
            current_best_value = self.expected_value(current_best_choice)

            # create a of lists, where each list is the list of all islands without one of the choices from costben
            options_list = [set(self.island_list).difference({island}) for island in costben_choice]
            # also allow the costben choice as a starting option to improve on
            options_list.append(costben_choice)

            for options in options_list:
                # list the allowed islands from worst to best
                ordered_options_worst_to_best = [option for option in self.island_rankings if option in options]
                current_choice = []
                flag = 0
                while flag == 0:
                    # check current cost and remaining project_budget
                    current_expenditure = self.expected_cost(current_choice)
                    remaining_funds = self.budget - current_expenditure
                    try:
                        # choose the best ben/cost island that is still under project_budget
                        best_option = np.array(ordered_options_worst_to_best)[
                            self.islands[0, ordered_options_worst_to_best] < remaining_funds][-1]
                        # add to the current_choice list and remove from the list of options
                        current_choice.append(ordered_options_worst_to_best.pop(
                            ordered_options_worst_to_best.index(best_option)))
                    except IndexError:  # index error when none are under project_budget - exit loop
                        flag = 1
                expected_value = self.expected_value(current_choice)  # calculate the expected value
                if expected_value > current_best_value:  # store if it is better than any other explored options
                    current_best_value = expected_value
                    current_best_choice = copy.copy(current_choice)

            self.optimal_choice = current_best_choice
            return current_best_choice
        return self.optimal_choice

    def choose_cheap(self):  # order by cheapest, fill up project_budget
        if self.cheap_choice is None:
            islands = copy.copy(self.island_list)  # local list of islands
            islands_costs = copy.copy(self.estimates[0, :])  # local list of island costs
            islands_costs_array = np.array([islands, islands_costs])
            islands_costs_array = islands_costs_array[:, islands_costs_array[1, :].argsort()]  # sorted list cheap first
            islands_ranked_cheap = [int(val) for val in islands_costs_array[0, :]]
            flag = 0

            cheap_choice = []
            while flag == 0:
                cheap_choice.append(islands_ranked_cheap.pop(0))
                current_costs = self.estimates[0, cheap_choice]
                current_total_cost = np.sum(current_costs)
                if current_total_cost > self.budget:
                    cheap_choice.pop(-1)
                    flag = 1
                if len(islands_ranked_cheap) == 0:
                    flag = 1
            self.cheap_choice = cheap_choice
            return cheap_choice
        return self.cheap_choice

    def choose_good(self):  # order by best benefit, fill up project_budget
        if self.good_choice is None:
            islands = copy.copy(self.island_list)
            islands_benefit = copy.copy(self.estimates[1, :])
            islands_benefits_array = np.array([islands, islands_benefit])
            islands_benefits_array = islands_benefits_array[:, islands_benefits_array[1, :].argsort()]
            islands_ranked_good = [int(val) for val in islands_benefits_array[0, :]]
            flag = 0

            good_choice = []
            while flag == 0:
                good_choice.append(islands_ranked_good.pop(-1))
                current_costs = self.estimates[0, good_choice]
                current_total_costs = np.sum(current_costs)
                if current_total_costs > self.budget:
                    good_choice.pop(-1)
                    flag = 1
                if len(islands_ranked_good) == 0:
                    flag = 1
                self.good_choice = good_choice
            return good_choice
        return self.good_choice

    def __setattr__(self, key, value):
        if key is 'num_islands':
            if not isinstance(value, int):
                raise TypeError("num_islands must be an integer")
        try:
            if key is 'estimation_variance':
                if (value <= 0) or (value >= 0.25):
                    raise ValueError("estimation_variance must be between 0 and 0.25. "
                                     "The input value of {} is outside this".format(value))
        except TypeError:
            raise TypeError("estimation_variance must be a number between 0 and 0.25. "
                            "An input of type {} was supplied".format(type(value)))
        self.__dict__[key] = value
        return


class IslandInvasivesEnsemble:
    def __init__(self, num_realisations, num_islands, budget=200.,
                 cost_average=50., cost_variance=40.,
                 value_variance=.1, estimation_variance=.2,
                 seed=3784123):

        self.num_realisations = num_realisations
        self.num_islands = num_islands
        self.budget = budget
        self.cost_average = cost_average
        self.cost_variance = cost_variance
        self.value_variance = value_variance
        self.estimation_variance = estimation_variance

        self.ensemble = None

        self.seed = seed
        np.random.seed(self.seed)

        self.analysis_complete = False

        self.random_expected_cost = None
        self.random_expected_value = None
        self.random_true_cost = None
        self.random_true_value = None

        self.costben_expected_cost = None
        self.costben_expected_value = None
        self.costben_true_cost = None
        self.costben_true_value = None

        self.optimal_expected_cost = None
        self.optimal_expected_value = None
        self.optimal_true_cost = None
        self.optimal_true_value = None

        self.cheap_expected_cost = None
        self.cheap_expected_value = None
        self.cheap_true_cost = None
        self.cheap_true_value = None

        self.good_expected_cost = None
        self.good_expected_value = None
        self.good_true_cost = None
        self.good_true_value = None

        self.force_run = False

    def generate_ensemble(self):
        if not self.force_run:
            try:
                result = pd.read_csv(self.file_save_string())

                self.random_expected_cost = result['random_expected_cost']
                self.random_true_cost = result['random_true_cost']
                self.random_expected_value = result['random_expected_value']
                self.random_true_value = result['random_true_value']

                self.costben_expected_cost = result['costben_expected_cost']
                self.costben_true_cost = result['costben_true_cost']
                self.costben_expected_value = result['costben_expected_value']
                self.costben_true_value = result['costben_true_value']

                self.optimal_expected_cost = result['optimal_expected_cost']
                self.optimal_true_cost = result['optimal_true_cost']
                self.optimal_expected_value = result['optimal_expected_value']
                self.optimal_true_value = result['optimal_true_value']

                self.cheap_expected_cost = result['cheap_expected_cost']
                self.cheap_true_cost = result['cheap_true_cost']
                self.cheap_expected_value = result['cheap_expected_value']
                self.cheap_true_value = result['cheap_true_value']

                self.good_expected_cost = result['good_expected_cost']
                self.good_true_cost = result['good_true_cost']
                self.good_expected_value = result['good_expected_value']
                self.good_true_value = result['good_true_value']

                self.analysis_complete = True
                return
            except:
                print('no file')
        if self.ensemble is None:
            ensemble = [IslandInvasives(self.num_islands, budget=self.budget,
                                        cost_average=self.cost_average, cost_variance=self.cost_variance,
                                        value_variance=self.value_variance,
                                        estimation_variance=self.estimation_variance,
                                        ) for i in range(self.num_realisations)]
            for island in ensemble:
                island.run_prioritisation()
            self.ensemble = ensemble
            return ensemble
        return self.ensemble

    def run_analysis(self):
        if not self.analysis_complete:
            while self.analysis_complete is False:
                self.store_costben_result()
                self.store_optimal_result()
                self.store_random_result()
                self.store_cheap_result()
                self.store_good_result()
                self.analysis_complete, bad_index = self.check_non_zero_data()
                if self.analysis_complete:
                    return
                else:
                    for index in list(bad_index):
                        print("Bad index: {}".format(index))
                        self.ensemble[index] = IslandInvasives(self.num_islands, budget=self.budget,
                                                               cost_average=self.cost_average,
                                                               cost_variance=self.cost_variance,
                                                               value_variance=self.value_variance,
                                                               estimation_variance=self.estimation_variance)
                        self.ensemble[index].run_prioritisation()

    # check that all data is non-zero
    def check_non_zero_data(self):
        bad_index = set()
        data_to_check = [self.random_expected_cost, self.random_expected_value,
                         self.random_true_cost, self.random_true_value,
                         self.costben_expected_cost, self.costben_expected_value,
                         self.costben_true_cost, self.costben_true_value,
                         self.optimal_expected_cost, self.optimal_expected_value,
                         self.optimal_true_cost, self.optimal_true_value,
                         self.cheap_expected_cost, self.cheap_expected_value,
                         self.cheap_true_cost, self.cheap_true_value,
                         self.good_expected_cost, self.cheap_expected_value,
                         self.good_true_cost, self.good_true_value]
        for data in data_to_check:
            try:
                bad_index.add(data.index(0))  # store index of each zero
            except ValueError:  # ignore value errors, as these are when the data is non-zero
                pass
        if len(bad_index) == 0:
            return True, bad_index  # with empty bad_index, return true
        else:
            return False, bad_index  # with bad_index non empty, return false, along with the indices

    def store_random_result(self):
        random_choices = [realisation.random_choice for realisation in self.ensemble]
        self.random_expected_cost, self.random_expected_value, self.random_true_cost, self.random_true_value \
            = self.return_cost_value(random_choices)

    def store_costben_result(self):
        costben_choices = [realisation.costben_choice for realisation in self.ensemble]
        self.costben_expected_cost, self.costben_expected_value, self.costben_true_cost, self.costben_true_value \
            = self.return_cost_value(costben_choices)

    def store_optimal_result(self):
        optimal_choices = [realisation.optimal_choice for realisation in self.ensemble]
        self.optimal_expected_cost, self.optimal_expected_value, self.optimal_true_cost, self.optimal_true_value \
            = self.return_cost_value(optimal_choices)

    def store_cheap_result(self):
        cheap_choice = [realisation.cheap_choice for realisation in self.ensemble]
        self.cheap_expected_cost, self.cheap_expected_value, self.cheap_true_cost, self.cheap_true_value \
            = self.return_cost_value(cheap_choice)

    def store_good_result(self):
        good_choice = [realisation.good_choice for realisation in self.ensemble]
        self.good_expected_cost, self.good_expected_value, self.good_true_cost, self.good_true_value = \
            self.return_cost_value(good_choice)

    def return_cost_value(self, choice):
        expected_cost = [r.expected_cost(c) for r, c in zip(self.ensemble, choice)]
        expected_value = [r.expected_value(c) for r, c in zip(self.ensemble, choice)]
        true_cost = [r.true_cost(c) for r, c in zip(self.ensemble, choice)]
        true_value = [r.true_value(c) for r, c in zip(self.ensemble, choice)]
        return expected_cost, expected_value, true_cost, true_value

    def _create_plot(self):
        if self.ensemble is None:
            self.generate_ensemble()
        if not self.analysis_complete:
            self.run_analysis()
        plt.subplot(2, 5, 1)
        self.scatter_plot(self.random_true_value, self.random_expected_value)
        plt.title('Random')
        plt.ylabel('True value')
        plt.subplot(2, 5, 6)
        self.scatter_plot(self.random_true_cost, self.random_expected_cost)
        plt.xlabel('Expected')
        plt.ylabel('True cost')

        plt.subplot(2, 5, 2)
        self.scatter_plot(self.costben_true_value, self.costben_expected_value)
        plt.title('Cost-benefit')
        plt.subplot(2, 5, 7)
        self.scatter_plot(self.costben_true_cost, self.costben_expected_cost)
        plt.xlabel('Expected')

        plt.subplot(2, 5, 3)
        self.scatter_plot(self.optimal_true_value, self.optimal_expected_value)
        plt.title('Optimal')
        plt.subplot(2, 5, 8)
        self.scatter_plot(self.optimal_true_cost, self.optimal_expected_cost)
        plt.xlabel('Expected')

        plt.subplot(2, 5, 4)
        self.scatter_plot(self.cheap_true_value, self.cheap_expected_value)
        plt.title('Cheap')
        plt.subplot(2, 5, 9)
        self.scatter_plot(self.cheap_true_cost, self.cheap_expected_cost)

        plt.subplot(2, 5, 5)
        self.scatter_plot(self.good_true_value, self.good_expected_value)
        plt.title('Good')
        plt.subplot(2, 5, 10)
        self.scatter_plot(self.good_true_cost, self.good_expected_cost)

    def show_results_plot(self):
        self._create_plot()
        plt.show()

    def save_plot(self, fname=None):
        self._create_plot()
        if fname is None:
            fname = self.parameter_name()
        plt.savefig('results/{}/plot_{}.png'.format(self.seed, fname))
        plt.close()

    def save_data(self, fname=None):
        if self.force_run is False:
            try:
                pd.read_csv(self.file_save_string())
                return
            except:
                pass
        if not self.analysis_complete:
            self.run_analysis()
        data = {'random_expected_cost': self.random_expected_cost,
                'random_true_cost': self.random_true_cost,
                'random_expected_value': self.random_expected_value,
                'random_true_value': self.random_true_value,
                'costben_expected_cost': self.costben_expected_cost,
                'costben_true_cost': self.costben_true_cost,
                'costben_expected_value': self.costben_expected_value,
                'costben_true_value': self.costben_true_value,
                'optimal_expected_cost': self.optimal_expected_cost,
                'optimal_true_cost': self.optimal_true_cost,
                'optimal_expected_value': self.optimal_expected_value,
                'optimal_true_value': self.optimal_true_value,
                'cheap_expected_cost': self.cheap_expected_cost,
                'cheap_true_cost': self.cheap_true_cost,
                'cheap_expected_value': self.cheap_expected_value,
                'cheap_true_value': self.cheap_true_value,
                'good_expected_cost': self.good_expected_cost,
                'good_true_cost': self.good_true_cost,
                'good_expected_value': self.good_expected_value,
                'good_true_value': self.good_true_value}
        df = pd.DataFrame(data)
        df.to_csv(self.file_save_string(fname))

    def file_save_string(self, fname=None):
        if fname is None:
            fname = self.parameter_name()
        return 'results/{}/data_{}.csv'.format(self.seed, fname)

    def parameter_name(self):
        return "num_reps{}_num_isl{}_estvar{}_budget{}_cost{}_var{}".format(
            self.num_realisations, self.num_islands, self.estimation_variance,
            self.budget, self.cost_average, self.cost_variance
        )

    @staticmethod
    def calc_surprise(true, expected):
        return np.mean(np.array(expected) - np.array(true))

    @staticmethod
    def scatter_plot(y, x):
        max_value = np.ceil(0.25+np.max(x+y))

        plt.plot([0, max_value], [0, max_value], 'k--')
        plt.scatter(x, y)
        plt.scatter(np.mean(x), np.mean(y), color=(0, 0, 0))

        plt.xlim([0, max_value])
        plt.ylim([0, max_value])


def plot_histogram_estimation_variance(var=0.01):
    draws = IslandInvasives(4, estimation_variance=var).beta_estimate_draws(50000)
    print(2*np.std(draws))
    plt.hist(draws)
    plt.show()
    return


def run_multiple_uncertainty(num_realisations=200, num_islands=50,
                             estimation_variance_list_input=(.02, .01), budget=5e6,
                             cost_ave=7e5, cost_var=1e11, seed=3784123):
    if str(seed) not in os.listdir('results'):
        os.mkdir('results/{}'.format(seed))
    for var in estimation_variance_list_input:
        print("Running estimation_variance = {}".format(var))
        ensemble = IslandInvasivesEnsemble(num_realisations, num_islands,
                                           estimation_variance=var, budget=budget,
                                           cost_average=cost_ave, cost_variance=cost_var, seed=seed)
        ensemble.generate_ensemble()
        ensemble.save_data()
        ensemble.save_plot()


def basic_plots(num_realisations=200, num_islands=50,
                estimation_variance_list_input=(.02, .01), budget=5e6,
                cost_ave=7e5, cost_var=1e11, seed = 3784123):
    fname = 'estvar{}{}_num_reps{}_num_isl{}_budget{}_cost{}_costvar{}'.format(estimation_variance_list_input[0],
                                                                               estimation_variance_list_input[-1],
                                                                               num_realisations, num_islands,
                                                                               budget, cost_ave, cost_var)
    opt_cost_surprise_list = []
    cb_cost_surprise_list = []
    rand_cost_surprise_list = []
    cheap_cost_surprise_list = []
    good_cost_surprise_list = []

    opt_val_surprise_list = []
    cb_val_surprise_list = []
    rand_val_surprise_list = []
    cheap_val_surprise_list = []
    good_val_surprise_list = []

    for var in estimation_variance_list_input:
        ensemble = IslandInvasivesEnsemble(num_realisations, num_islands,
                                           estimation_variance=var, budget=budget,
                                           cost_average=cost_ave, cost_variance=cost_var)
        ensemble.generate_ensemble()
        opt_cost_surprise = ensemble.optimal_true_cost - ensemble.optimal_expected_cost
        cb_cost_surprise = ensemble.costben_true_cost - ensemble.costben_expected_cost
        rand_cost_surprise = ensemble.random_true_cost - ensemble.random_expected_cost
        cheap_cost_surprise = ensemble.cheap_true_cost - ensemble.cheap_expected_cost
        good_cost_surprise = ensemble.good_true_cost - ensemble.good_expected_cost

        opt_val_surprise = ensemble.optimal_true_value - ensemble.optimal_expected_value
        cb_val_surprise = ensemble.costben_true_value - ensemble.costben_expected_value
        rand_val_surprise = ensemble.random_true_value - ensemble.random_expected_value
        cheap_val_surprise = ensemble.cheap_true_value - ensemble.cheap_expected_value
        good_val_surprise = ensemble.good_true_value - ensemble.good_expected_value

        opt_cost_surprise_list.append([np.mean(opt_cost_surprise)/np.mean(ensemble.optimal_true_cost),
                                       np.std(opt_cost_surprise)])
        cb_cost_surprise_list.append([np.mean(cb_cost_surprise)/np.mean(ensemble.costben_true_cost),
                                      np.std(cb_cost_surprise)])
        rand_cost_surprise_list.append([np.mean(rand_cost_surprise)/np.mean(ensemble.random_true_cost),
                                        np.std(rand_cost_surprise)])
        cheap_cost_surprise_list.append([np.mean(cheap_cost_surprise)/np.mean(ensemble.cheap_true_cost),
                                         np.std(cheap_cost_surprise)])
        good_cost_surprise_list.append([np.mean(good_cost_surprise)/np.mean(ensemble.good_true_cost),
                                        np.std(good_cost_surprise)])

        opt_val_surprise_list.append([np.mean(opt_val_surprise)/np.mean(ensemble.optimal_true_value),
                                      np.std(opt_val_surprise)])
        cb_val_surprise_list.append([np.mean(cb_val_surprise)/np.mean(ensemble.costben_true_value),
                                     np.std(cb_val_surprise)])
        rand_val_surprise_list.append([np.mean(rand_val_surprise)/np.mean(ensemble.random_true_value),
                                       np.std(rand_val_surprise)])
        cheap_val_surprise_list.append([np.mean(cheap_val_surprise)/np.mean(ensemble.cheap_true_value),
                                        np.std(cheap_val_surprise)])
        good_val_surprise_list.append([np.mean(good_val_surprise)/np.mean(ensemble.good_true_value),
                                      np.std(good_val_surprise)])

    opt_cost_surprise_array = np.array(opt_cost_surprise_list)
    cb_cost_surprise_array = np.array(cb_cost_surprise_list)
    rand_cost_surprise_array = np.array(rand_cost_surprise_list)
    cheap_cost_surprise_array = np.array(cheap_cost_surprise_list)
    good_cost_surprise_array = np.array(good_cost_surprise_list)

    estimation_variance_list_input = 100 * np.sqrt(estimation_variance_list_input)
    plt.plot(estimation_variance_list_input, 100 * opt_cost_surprise_array[:, 0])
    plt.plot(estimation_variance_list_input, 100 * cb_cost_surprise_array[:, 0])
    plt.plot(estimation_variance_list_input, 100 * rand_cost_surprise_array[:, 0])
    plt.plot(estimation_variance_list_input, 100 * cheap_cost_surprise_array[:, 0])
    plt.plot(estimation_variance_list_input, 100 * good_cost_surprise_array[:, 0])
    plt.legend(['Optimal', 'Cost-benefit', 'Random', 'Cheap', 'Good'])
    plt.xlabel('Measurement error %')
    plt.ylabel('% cost surprise')

    plt.savefig('results/{}/cost_surprise_{}.png'.format(seed, fname))

    plt.close()

    opt_val_surprise_array = np.array(opt_val_surprise_list)
    cb_val_surprise_array = np.array(cb_val_surprise_list)
    rand_val_surprise_array = np.array(rand_val_surprise_list)
    cheap_val_surprise_array = np.array(cheap_val_surprise_list)
    good_val_surprise_array = np.array(good_val_surprise_list)

    plt.plot(estimation_variance_list_input, 100 * opt_val_surprise_array[:, 0])
    plt.plot(estimation_variance_list_input, 100 * cb_val_surprise_array[:, 0])
    plt.plot(estimation_variance_list_input, 100 * rand_val_surprise_array[:, 0])
    plt.plot(estimation_variance_list_input, 100 * cheap_val_surprise_array[:, 0])
    plt.plot(estimation_variance_list_input, 100 * good_val_surprise_array[:, 0])
    plt.legend(['Optimal', 'Cost-benefit', 'Random', 'Cheap', 'Good'])
    plt.xlabel('Measurement error %')
    plt.ylabel('% value surprise')

    plt.savefig('results/{}/value_surprise_{}.png'.format(seed, fname))

    plt.close()


if __name__ == "__main__":
    # cost parameters - this produces costs between approx 250k and 1.5mil
    project_cost_average = 7e5
    project_cost_variance = 1e11
    project_budget = 5e6
    repeats = 1000

    estimation_variance_list = list(np.arange(.005, .1, .005)) + [.1]

    # set the simulation_seed - result plots are using 3784123,
    # the other one is to check the results aren't specific to the simulation_seed choice
    simulation_seed = 3784123
    # simulation_seed = 2489012

    # The following line makes the list clean (ie 0.09 instead of 0.09000000000000001) for file naming purposes
    estimation_variance_list = np.round(estimation_variance_list, decimals=3)

    # run the simulations across the range of estimation variances
    run_multiple_uncertainty(num_realisations=repeats, num_islands=150,
                             estimation_variance_list_input=estimation_variance_list,
                             budget=project_budget, cost_ave=project_cost_average, cost_var=project_cost_variance,
                             seed=simulation_seed)

    # create and save basic plots of the simulation data
    # these are not the final plots used in the paper
    basic_plots(num_realisations=repeats, num_islands=150,
                estimation_variance_list_input=estimation_variance_list,
                budget=project_budget, cost_ave=project_cost_average, cost_var=project_cost_variance,
                seed=simulation_seed)
