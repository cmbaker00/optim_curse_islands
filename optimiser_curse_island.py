import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import pandas as pd
from functools import lru_cache


class IslandInvasives:
    def __init__(self, num_islands, budget = 200., cost_average = 50., cost_variance = 40.,
                 value_variance = .1, estimation_variance = .05):
        self.num_islands = num_islands
        self.island_list = list(range(self.num_islands))

        self.cost_average = cost_average #average cost to remove species
        self.cost_variance = cost_variance #variance in cost
        self.cost_distribution = "LogNormal" #distribution to use (Uniform or LogNormal)

        self.value_average = 1 #value of removing species
        self.value_variance = value_variance #variance in values
        self.value_distribution = "LogNormal" #distribution to use (Uniform or LogNormal)

        self.estimation_variance = estimation_variance #variance in estimating cost & value

        self.budget = budget #total budget

        self.islands = None #true values about islands
        self.estimates = None   #estimated values
        self.island_rankings = None #Ranking of islands worst to best.

        #placeholders for the list of choices
        self.random_choice = None
        self.costben_choice = None
        self.optimal_choice = None

    def run_prioritisation(self): #run the full analysis for a set of islands
        self.generate_islands()
        self.generate_estimates()
        self.choose_random()
        self.choose_costben()
        self.choose_optimal()


    def generate_islands(self): #generate island parameters
        if self.islands is None:
            island_costs = self.generate_costs()
            island_values = self.generate_values()
            #store the costs and values in an array
            self.islands = np.array([island_costs, island_values, island_values/island_costs])
        return


    #choose a random set of islans within budget
    def choose_random(self):
        if self.random_choice is None:
            flag = 0
            islands = copy.copy(self.island_list) #create a local island list
            random_choice = [] #empty list to be populated with the set of islands
            while flag == 0:
                #add a random island to the choice list and remove it from the local island list
                random_choice.append(islands.pop(islands.index(np.random.choice(islands))))

                current_costs = self.estimates[0,random_choice]
                current_total_cost = np.sum(current_costs)

                if current_total_cost > self.budget: #check is overbudgdet
                    random_choice.pop(-1) #remove the last choice that went overbudget
                    flag = 1
                if len(islands) == 0:
                    flag = 1 #exit if all islands are chosen

            self.random_choice = random_choice
            return random_choice

        return self.random_choice

    #choose islands by cost vs benefit
    def choose_costben(self):
        if self.costben_choice is None:
            islands = copy.copy(self.island_list) #local island list
            islands_bencost = copy.copy(self.estimates[2,:]) #local list of benefit/cost for ech island
            islands_bencost_array = np.array([islands, islands_bencost]) #join them together
            islands_bencost_array = islands_bencost_array[:,islands_bencost_array[1,:].argsort()] #sort by benefit/cost

            flag = 0
            islands_ranked_worst_to_best = [int(val) for val in islands_bencost_array[0,:]] #return a list of islands ranked from worst to best
            self.island_rankings = copy.copy(islands_ranked_worst_to_best) #save the list for use elsewhere
            costben_choice = []
            while flag == 0: #as per random choice, but always adding the next best island until the budget is exhausted
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
            costben_choice = self.choose_costben() # start with the costben solution and explore 'nearby' options
            current_best_choice = costben_choice
            current_best_value = self.expected_value(current_best_choice)

            #create a of lists, where each list is the list of all islands without one of the chocies from costben
            options_list = [set(self.island_list).difference({island}) for island in costben_choice]
            #also allow the costben choice as a starting option to imporve on
            options_list.append(costben_choice)

            for options in options_list:
                # list the allowed islands from worst to best
                ordered_options_worst_to_best = [option for option in self.island_rankings if option in options]
                current_choice = []
                flag = 0
                while flag == 0:
                    #check current cost and remaining budget
                    current_expenditure = self.expected_cost(current_choice)
                    remaining_funds = self.budget - current_expenditure
                    try:
                        #choose the best ben/cost island that is still under budget
                        best_option = np.array(ordered_options_worst_to_best)[
                            self.islands[0, ordered_options_worst_to_best] < remaining_funds][-1]
                        #add to the current_choice list and remove from the list of options
                        current_choice.append(ordered_options_worst_to_best.pop(ordered_options_worst_to_best.index(best_option)))
                    except IndexError: #index error when none are under budget - exit loop
                        flag = 1
                expected_value = self.expected_value(current_choice) #calc the expected value
                if expected_value > current_best_value: #store if it is better than any other explored options
                    current_best_value = expected_value
                    current_best_choice = copy.copy(current_choice)

            self.optimal_choice = current_best_choice
            return current_best_choice
        return self.optimal_choice



    #The following four functions return the expected/true value/cost of an input list of islands
    def expected_value(self, choice):
        return np.sum(self.estimates[1, choice])

    def expected_cost(self, choice):
        return np.sum(self.estimates[0, choice])

    def true_value(self, choice):
        return np.sum(self.islands[1, choice])

    def true_cost(self, choice):
        return np.sum(self.islands[0, choice])

    #generate and store estimates of cost and value
    def generate_estimates(self):
        if self.estimates is None:
            if self.islands is None:
                raise RuntimeError("generate_islands must be called before generate estimtes")

            estimates = self.islands*self.beta_estimate_draws(self.islands.shape)
            if (estimates <= 0).any():
                raise ValueError('All value/cost estimates must be positive')
            estimates[2,:] = estimates[1,:]/estimates[0,:] #store a 3rd row, the benefit/cost of each island
            self.estimates = estimates
        return


    #generate estimates with a beta distribution with parameters Beta(a,a)
    #Variance of beta distribution is 1/(4+8a), so a = (1-4var)/(8var). Variance must be less than 0.25
    #However, the beta distribution is 'doubled' to have support [0,2], so thaat also doubles the varaiaacnes
    #Hence, variance must be halved at this calculate -> input variance can be up to 0.5
    def beta_estimate_draws(self, num):
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
        #I couldn't find an equation for mu and sigma as a function of
        #mean and variance for the lognormal distribution
        #this uses fixed point iteration
        mu = np.log(mean) #initial guess of mu
        sigma = None
        flag = 0
        while flag == 0:
            sigma = np.sqrt(np.log(0.5 * (1 + np.sqrt(1 + 4 * variance * np.exp(-2 * mu))))) #true sigma, given current mu
            prev_mu = copy.copy(mu)
            mu = np.log(mean) - (sigma ** 2) / 2 #update mu.
            if np.abs(prev_mu - mu) < .001:
                flag = 1
            if np.abs(mu) > 1e6:
                raise Exception
        return mu, sigma


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
    def __init__(self, num_realisations, num_islands, budget = 200.,
                 cost_average = 50., cost_variance = 40.,
                 value_variance = .1, estimation_variance = .2,
                 seed = 3784123):

        self.num_realisations = num_realisations
        self.num_islands = num_islands
        self.budget = budget
        self.cost_average = cost_average
        self.cost_variance = cost_variance
        self.value_variance = value_variance
        self.estimation_variance = estimation_variance

        self.ensemble = None

        np.random.seed(seed)

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

        self.force_run = False

    def generate_ensemble(self):
        if not self.force_run:
            try:
                result = pd.read_csv(self.file_save_string())
                self.analysis_complete = True

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
                return
            except:
                pass
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
            self.store_costben_result()
            self.store_optimal_result()
            self.store_random_result()
            self.analysis_complete = True
        return


    def store_random_result(self):
        random_choices = [realisation.random_choice for realisation in self.ensemble]
        self.random_expected_cost, self.random_expected_value, self.random_true_cost, self.random_true_value = self.return_cost_value(random_choices)


    def store_costben_result(self):
        costben_choices = [realisation.costben_choice for realisation in self.ensemble]
        self.costben_expected_cost, self.costben_expected_value, self.costben_true_cost, self.costben_true_value = self.return_cost_value(costben_choices)


    def store_optimal_result(self):
        optimal_choices = [realisation.optimal_choice for realisation in self.ensemble]
        self.optimal_expected_cost, self.optimal_expected_value, self.optimal_true_cost, self.optimal_true_value = self.return_cost_value(optimal_choices)


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
        plt.subplot(231)
        self.scatter_plot(self.random_true_value, self.random_expected_value)
        plt.title('Random')
        plt.ylabel('True value')
        plt.subplot(234)
        self.scatter_plot(self.random_true_cost, self.random_expected_cost)
        plt.xlabel('Expected')
        plt.ylabel('True cost')

        plt.subplot(232)
        self.scatter_plot(self.costben_true_value, self.costben_expected_value)
        plt.title('Cost-benefit')
        plt.subplot(235)
        self.scatter_plot(self.costben_true_cost, self.costben_expected_cost)
        plt.xlabel('Expected')

        plt.subplot(233)
        self.scatter_plot(self.optimal_true_value, self.optimal_expected_value)
        plt.title('Optimal')
        plt.subplot(236)
        self.scatter_plot(self.optimal_true_cost, self.optimal_expected_cost)
        plt.xlabel('Expected')


    def show_results_plot(self):
        self._create_plot()
        plt.show()


    def save_plot(self, fname = None):
        self._create_plot()
        if fname is None:
            fname = self.parameter_name()
        plt.savefig('results/plot_{}.png'.format(fname))
        plt.close()

    def save_data(self, fname = None):
        if self.force_run is False:
            try:
                result = pd.read_csv(self.file_save_string())
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
                'optimal_true_value': self.optimal_true_value}
        df = pd.DataFrame(data)
        df.to_csv(self.file_save_string(fname))


    def file_save_string(self, fname = None):
        if fname is None:
            fname = self.parameter_name()
        return 'results/data_{}.csv'.format(fname)


    def parameter_name(self):
        return "num_reps{}_num_isl{}_estvar{}_budget{}_cost{}_var{}".format(
            self.num_realisations, self.num_islands, self.estimation_variance,
            self.budget, self.cost_average, self.cost_variance
        )

    @staticmethod
    def calc_surprise(true, expected):
        return np.mean(np.array(expected) - np.array(true))


    @staticmethod
    def scatter_plot(y,x):
        max_value = np.ceil(0.25+np.max(x+y))

        plt.plot([0,max_value],[0,max_value],'k--')
        plt.scatter(x,y)
        plt.scatter(np.mean(x), np.mean(y),color=(0,0,0))

        plt.xlim([0,max_value])
        plt.ylim([0,max_value])


def plot_histogram_estimation_variance(var=0.01):
    draws = IslandInvasives(4, estimation_variance=var).beta_estimate_draws(50000)
    print(2*np.std(draws))
    plt.hist(draws)
    plt.show()
    return


def run_multiple_uncertainty(num_realisations=200,num_islands=50,
                                       estimation_variance_list = (.02, .01), budget=5e6,
                                       cost_ave = 7e5, cost_var = 1e11):
    for var in estimation_variance_list:
        print("Running estimation_variance = {}".format(var))
        ensemble = IslandInvasivesEnsemble(num_realisations, num_islands,
                                           estimation_variance=var, budget=budget,
                                           cost_average=cost_ave, cost_variance=cost_var)
        ensemble.generate_ensemble()
        ensemble.save_data()
        ensemble.save_plot()



def basic_plots(num_realisations=200,num_islands=50,
                                       estimation_variance_list = (.02, .01), budget=5e6,
                                       cost_ave = 7e5, cost_var = 1e11):
    fname = 'estvar{}{}_num_reps{}_num_isl{}_budget{}_cost{}_costvar{}'.format(estimation_variance_list[0],
                                                                               estimation_variance_list[-1],
                                                                               num_realisations,num_islands,
                                                          budget, cost_ave, cost_var)

    opt_cost_surprise_list = []
    cb_cost_surprise_list = []
    rand_cost_surprise_list = []
    opt_val_surprise_list = []
    cb_val_surprise_list = []
    rand_val_surprise_list = []
    for var in estimation_variance_list:
        ensemble = IslandInvasivesEnsemble(num_realisations, num_islands,
                                           estimation_variance=var, budget=budget,
                                           cost_average=cost_ave, cost_variance=cost_var)
        ensemble.generate_ensemble()
        opt_cost_surprise = ensemble.optimal_true_cost - ensemble.optimal_expected_cost
        cb_cost_surprise = ensemble.costben_true_cost - ensemble.costben_expected_cost
        rand_cost_surprise = ensemble.random_true_cost - ensemble.random_expected_cost

        opt_val_surprise = ensemble.optimal_true_value - ensemble.optimal_expected_value
        cb_val_surprise = ensemble.costben_true_value - ensemble.costben_expected_value
        rand_val_surprise = ensemble.random_true_value - ensemble.random_expected_value

        opt_cost_surprise_list.append([np.mean(opt_cost_surprise)/np.mean(ensemble.optimal_true_cost),
                                       np.std(opt_cost_surprise)])
        cb_cost_surprise_list.append([np.mean(cb_cost_surprise)/np.mean(ensemble.costben_true_cost),
                                      np.std(cb_cost_surprise)])
        rand_cost_surprise_list.append([np.mean(rand_cost_surprise)/np.mean(ensemble.random_true_cost),
                                        np.std(rand_cost_surprise)])

        opt_val_surprise_list.append([np.mean(opt_val_surprise)/np.mean(ensemble.optimal_true_value),
                                       np.std(opt_val_surprise)])
        cb_val_surprise_list.append([np.mean(cb_val_surprise)/np.mean(ensemble.costben_true_value),
                                      np.std(cb_val_surprise)])
        rand_val_surprise_list.append([np.mean(rand_val_surprise)/np.mean(ensemble.random_true_value),
                                        np.std(rand_val_surprise)])

    opt_cost_surprise_array = np.array(opt_cost_surprise_list)
    cb_cost_surprise_array = np.array(cb_cost_surprise_list)
    rand_cost_surprise_array = np.array(rand_cost_surprise_list)

    estimation_variance_list = 100*np.sqrt(estimation_variance_list)
    plt.plot(estimation_variance_list, 100*opt_cost_surprise_array[:,0])
    plt.plot(estimation_variance_list, 100*cb_cost_surprise_array[:,0])
    plt.plot(estimation_variance_list, 100*rand_cost_surprise_array[:,0])
    plt.legend(['Optimal', 'Cost-benefit', 'Random'])
    plt.xlabel('Measurement error %')
    plt.ylabel('% cost surprise')


    plt.savefig('results/cost_surprise_{}.png'.format(fname))

    plt.close()

    opt_val_surprise_array = np.array(opt_val_surprise_list)
    cb_val_surprise_array = np.array(cb_val_surprise_list)
    rand_val_surprise_array = np.array(rand_val_surprise_list)

    plt.plot(estimation_variance_list, 100*opt_val_surprise_array[:,0])
    plt.plot(estimation_variance_list, 100*cb_val_surprise_array[:,0])
    plt.plot(estimation_variance_list, 100*rand_val_surprise_array[:,0])
    plt.legend(['Optimal', 'Cost-benefit', 'Random'])
    plt.xlabel('Measurement error %')
    plt.ylabel('% value surprise')

    plt.savefig('results/value_surprise_{}.png'.format(fname))

    plt.close()


if  __name__ == "__main__":
    # cost parameters - this produces costs between approx 250k and 1.5mil
    cost_average = 7e5
    cost_variance = 1e11
    budget = 5e6
    repeats = 1000

    # estimation_variance_list = [.05, .04,.03,.02,.01,.005,.001]
    estimation_variance_list = list(np.arange(.1,.01,-.01))+list(np.arange(.01,.001,-.001))+[.001]
    estimation_variance_list = np.round(estimation_variance_list, decimals=3)
    # run_multiple_uncertainty(num_realisations=5000, num_islands=150,
    #                          estimation_variance_list=estimation_variance_list,
    #                          budget=budget, cost_ave=cost_average, cost_var=cost_variance)

    basic_plots(num_realisations=repeats, num_islands=150,
                             estimation_variance_list=estimation_variance_list,
                             budget=budget, cost_ave=cost_average, cost_var=cost_variance)



    #
    # estimation_variance = 0.02
    # plot_histogram_estimation_variance(estimation_variance)
    #
    # ensemble = IslandInvasivesEnsemble(num_realisations=200,num_islands=50,
    #                                    estimation_variance=estimation_variance, budget=5e6,
    #                                    cost_average=cost_average, cost_variance=cost_variance)
    # ensemble.generate_ensemble()
    # ensemble.show_results_plot()
    # ensemble.save_data()
    # ensemble.save_plot()







    # tstart = time.time()
    # for i in range(100):
    #     sys = IslandInvasives(400, budget=500)
    #     # sys.generate_islands()
    #     # sys.generate_estimates()
    #     sys.choose_optimal()
    #     rand_choice = sys.choose_random()
    #     costben_choice = sys.choose_costben()
    #     optimal_choice = sys.choose_optimal()
    #     # print(sys.expected_cost(rand_choice))
    #     # print(sys.true_cost(rand_choice))
    # print(sys.expected_cost(costben_choice),sys.true_cost(costben_choice))
    # print(sys.expected_value(costben_choice),sys.true_value(costben_choice))
    # print(sys.expected_cost(optimal_choice),sys.true_cost(optimal_choice))
    # print(sys.expected_value(optimal_choice),sys.true_value(optimal_choice))
    # tend = time.time()
    # print(tend-tstart)
