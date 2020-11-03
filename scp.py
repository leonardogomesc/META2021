# Leonardo Gomes Capozzi - up201503708

import random
import time
import numpy as np
import matplotlib.pyplot as plt
import math

seed = 10


class Instance:

    def __init__(self, filename, best_solution):
        file = open(filename, 'r')

        size = file.readline().split()

        total_elems = int(size[0])
        total_subsets = int(size[1])

        universe = [i for i in range(total_elems)]
        subset_universe = [i for i in range(total_subsets)]
        subset_weights = []
        subsets = [[] for _ in range(total_subsets)]
        elem_subsets = []

        while len(subset_weights) < total_subsets:
            line = file.readline().split()

            for sw in line:
                subset_weights.append(int(sw))

        curr_elem = 0

        while curr_elem < total_elems:
            n_subsets = int(file.readline().split()[0])

            elem_subset = []

            while n_subsets > 0:
                line = file.readline().split()

                for s in line:
                    subsets[int(s) - 1].append(curr_elem)
                    elem_subset.append(int(s) - 1)
                    n_subsets -= 1

            elem_subsets.append(elem_subset)

            curr_elem += 1

        file.close()

        self.universe = universe
        self.subset_universe = subset_universe
        self.subset_weights = subset_weights
        self.subsets = subsets
        self.elem_subsets = elem_subsets
        self.best_solution = best_solution

        self.universe_set = set(self.universe)
        self.subset_universe_set = set(self.subset_universe)

        subsets_set = []

        for s in self.subsets:
            subsets_set.append(set(s))

        self.subsets_set = subsets_set

        matrix = np.zeros((len(self.subsets), len(self.elem_subsets)))

        for s in range(matrix.shape[0]):
            for e in self.subsets[s]:
                matrix[s][e] = 1

        self.matrix = matrix

    def calculate_cost(self, selected_subsets):
        cost = 0

        for s in selected_subsets:
            cost += self.subset_weights[s]

        return cost

    def calculate_percentage_deviation(self, selected_subsets):
        cost = self.calculate_cost(selected_subsets)

        pd = (abs(cost - self.best_solution) * 100) / self.best_solution

        return pd

    def calculate_element_count(self, selected_subsets):
        element_count = np.zeros(self.matrix.shape[1])

        for s in selected_subsets:
            element_count = element_count + self.matrix[s]

        return element_count


def remove_redundant(instance, selected_subsets, element_count=None, return_extra=False, random_shuffle=False):
    # copy selected subsets
    selected_subsets = selected_subsets.copy()

    # sort by cost per element
    if random_shuffle:
        random.shuffle(selected_subsets)
    else:
        selected_subsets.sort(key=lambda e: instance.subset_weights[e]/len(instance.subsets[e]))

    # calculate or copy array containing the number of times each element has been selected
    if element_count is None:
        element_count = instance.calculate_element_count(selected_subsets)
    else:
        element_count = element_count.copy()

    # calculate cost
    cost = 0

    final_subsets = []

    i = len(selected_subsets) - 1

    while i >= 0:
        # try to remove subset
        element_count = element_count - instance.matrix[selected_subsets[i]]

        # check if there is an uncovered element
        if 0 in element_count:
            # add subset if there is an uncovered element
            element_count = element_count + instance.matrix[selected_subsets[i]]
            cost += instance.subset_weights[selected_subsets[i]]
            final_subsets.append(selected_subsets[i])

        i -= 1

    if return_extra:
        return final_subsets, cost, element_count

    return final_subsets


# choose subset with min heuristic value
def CH1_GR(instance, alpha=0.1):
    selected_subsets = []  # index of selected subsets
    uncovered_elements = instance.universe_set  # uncovered elements

    while len(uncovered_elements) > 0:
        # calculate subsets with at least one uncovered element
        aval_subsets = []

        for e in uncovered_elements:
            aval_subsets.extend(instance.elem_subsets[e])

        aval_subsets = list(set(aval_subsets))

        aval_subsets.sort(key=lambda s: instance.subset_weights[s] / len(instance.subsets_set[s].intersection(uncovered_elements)))

        n = max(1, int(alpha*len(aval_subsets)))

        subset = aval_subsets[random.randint(0, n-1)]

        # add subset to list
        selected_subsets.append(subset)

        # calculate uncovered elements
        uncovered_elements = uncovered_elements.difference(instance.subsets_set[subset])

    return selected_subsets


# choose subset with min heuristic value
def CH1(instance):
    selected_subsets = []  # index of selected subsets
    uncovered_elements = instance.universe_set  # uncovered elements

    while len(uncovered_elements) > 0:
        # calculate subsets with at least one uncovered element
        aval_subsets = []

        for e in uncovered_elements:
            aval_subsets.extend(instance.elem_subsets[e])

        aval_subsets = list(set(aval_subsets))

        # choose subset with min cost
        subset = min(aval_subsets, key=lambda s: instance.subset_weights[s] / len(instance.subsets_set[s].intersection(uncovered_elements)))

        # add subset to list
        selected_subsets.append(subset)

        # calculate uncovered elements
        uncovered_elements = uncovered_elements.difference(instance.subsets_set[subset])

    return selected_subsets


# choose element with min num of subsets containing it then subset with min heuristic value
def CH2(instance):
    selected_subsets = []  # index of selected subsets
    uncovered_elements = instance.universe_set  # uncovered elements

    while len(uncovered_elements) > 0:
        # calculate uncovered element with lowest amount of subsets containing it
        elem = min(uncovered_elements, key=lambda e: len(instance.elem_subsets[e]))

        # choose subset containing elem with the lowest cost
        subset = min(instance.elem_subsets[elem], key=lambda s: instance.subset_weights[s])

        # add subset to list
        selected_subsets.append(subset)

        # update uncovered elements
        uncovered_elements = uncovered_elements.difference(instance.subsets_set[subset])

    return selected_subsets


# random elem, choose subset with min cost
def CH3(instance):
    selected_subsets = []  # index of selected subsets
    uncovered_elements = instance.universe_set  # uncovered elements

    while len(uncovered_elements) > 0:
        # select random uncovered element
        random.seed(seed)
        rand_elem = random.sample(uncovered_elements, 1)[0]

        # choose subset containing elem with the lowest cost
        subset = min(instance.elem_subsets[rand_elem], key=lambda s: instance.subset_weights[s])

        # add subset to list
        selected_subsets.append(subset)

        # update uncovered elements
        uncovered_elements = uncovered_elements.difference(instance.subsets_set[subset])

    return selected_subsets


# random
def CHR(instance):
    selected_subsets = []  # index of selected subsets
    uncovered_elements = instance.universe_set  # uncovered elements

    while len(uncovered_elements) > 0:
        # select random uncovered element
        # random.seed(seed)
        rand_elem = random.sample(uncovered_elements, 1)[0]

        # choose random subset containing elem
        # random.seed(seed)
        subset = random.choice(instance.elem_subsets[rand_elem])

        # add subset to list
        selected_subsets.append(subset)

        # update uncovered elements
        uncovered_elements = uncovered_elements.difference(instance.subsets_set[subset])

    return selected_subsets


def next_neighbour(instance, curr_sol, element_count=None):
    # copy solution list
    curr_sol = curr_sol.copy()

    # calculate list of subsets that are not in the current solution
    remaining_subsets = list(instance.subset_universe_set.difference(set(curr_sol)))

    # sort by cost per element
    remaining_subsets.sort(key=lambda e: instance.subset_weights[e]/len(instance.subsets[e]))

    # calculate array containing the number of times each element has been selected
    if element_count is None:
        element_count = instance.calculate_element_count(curr_sol)
    else:
        element_count.copy()

    for rs in remaining_subsets:
        # append a subset to the current solution
        curr_sol.append(rs)
        element_count = element_count + instance.matrix[rs]

        # remove redundancy
        yield remove_redundant(instance, curr_sol, element_count=element_count, return_extra=True)

        # remove the subset that was added
        curr_sol.pop()
        element_count = element_count - instance.matrix[rs]


def next_random_neighbour(instance, curr_sol, element_count=None, n_subsets=1):
    # copy solution list
    curr_sol = curr_sol.copy()

    # calculate list of subsets that are not in the current solution
    remaining_subsets = instance.subset_universe_set.difference(set(curr_sol))

    n_subsets = min(n_subsets, len(remaining_subsets))

    # calculate array containing the number of times each element has been selected
    if element_count is None:
        element_count = instance.calculate_element_count(curr_sol)
    else:
        element_count.copy()

    while True:
        new_subsets = random.sample(remaining_subsets, n_subsets)

        curr_sol.extend(new_subsets)
        count_addition = np.sum(instance.matrix[new_subsets], axis=0)
        element_count = element_count + count_addition

        yield remove_redundant(instance, curr_sol, element_count=element_count, return_extra=True, random_shuffle=True)

        del curr_sol[-n_subsets:]
        element_count = element_count - count_addition


def improvement_heuristic(instance, selected_subsets, first_improvement=True):
    curr_sol = selected_subsets.copy()  # current solution
    curr_sol_cost = instance.calculate_cost(curr_sol)  # current solution cost
    curr_element_count = instance.calculate_element_count(curr_sol)

    failed = False

    while not failed:
        failed = True

        for new_sol, new_sol_cost, new_element_count in next_neighbour(instance, curr_sol, element_count=curr_element_count):

            # check if new solution has a lower cost than current solution
            if new_sol_cost < curr_sol_cost:
                curr_sol = new_sol
                curr_sol_cost = new_sol_cost
                curr_element_count = new_element_count
                failed = False

                # if first improvement take a step
                if first_improvement:
                    break

    return curr_sol


def simulated_annealing(instance, selected_subsets):
    plt.ion()
    plt.show()

    data_x = []
    data_y = []

    hl = plt.plot(data_x, data_y)[0]

    #####

    iter_without_change = 0

    max_iter_without_change = 50
    fixed_iter = 20000
    n_subsets = 10

    temperature = 2
    cooling_ratio = 0.95

    timestep = 1

    curr_sol = selected_subsets.copy()  # current solution
    curr_sol_cost = instance.calculate_cost(curr_sol)  # current solution cost
    curr_element_count = instance.calculate_element_count(curr_sol)
    curr_iter = next_random_neighbour(instance, curr_sol, element_count=curr_element_count, n_subsets=n_subsets)

    global_sol = curr_sol
    global_sol_cost = curr_sol_cost

    data_x.append(0)
    data_y.append(curr_sol_cost)

    while iter_without_change < max_iter_without_change:
        iter_without_change += 1

        for _ in range(fixed_iter):
            new_sol, new_sol_cost, new_element_count = next(curr_iter)

            update_sol = False

            if new_sol_cost < curr_sol_cost:
                update_sol = True
            elif new_sol_cost == curr_sol_cost:
                if set(new_sol) != set(curr_sol):
                    update_sol = True
            else:
                p = math.exp(-(new_sol_cost-curr_sol_cost)/temperature)

                if random.random() < p:
                    update_sol = True

            if update_sol:
                if new_sol_cost < global_sol_cost:
                    global_sol = new_sol
                    global_sol_cost = new_sol_cost
                    iter_without_change = 0

                curr_sol = new_sol
                curr_sol_cost = new_sol_cost
                curr_element_count = new_element_count
                curr_iter = next_random_neighbour(instance, new_sol, element_count=new_element_count, n_subsets=n_subsets)

                data_x.append(timestep)
                data_y.append(new_sol_cost)

            timestep += 1

        temperature *= cooling_ratio

        hl.set_xdata(np.append(hl.get_xdata(), data_x))
        hl.set_ydata(np.append(hl.get_ydata(), data_y))

        ax = plt.gca()
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        data_x = []
        data_y = []

        print('\ntemp: ' + str(temperature))
        print(global_sol_cost)
        print(iter_without_change)

    print(timestep)

    return global_sol


def grasp(instance, max_iter, alpha):
    global_sol = None
    global_sol_cost = float('inf')

    for _ in range(max_iter):
        selected_subsets = CH1_GR(instance, alpha=alpha)
        selected_subsets = improvement_heuristic(instance, selected_subsets, first_improvement=True)

        cost = instance.calculate_cost(selected_subsets)

        if cost < global_sol_cost:
            global_sol = selected_subsets
            global_sol_cost = cost
            print(global_sol_cost)

    return global_sol


def assignment1(instance_objs):
    print('Assignment 1\n')

    # CH1
    start = time.time()

    pd = []
    pd_RE = []

    for instance in instance_objs:
        selected_subsets = CH1(instance)
        selected_subsets_RE = remove_redundant(instance, selected_subsets)

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)
        percentage_deviation_RE = instance.calculate_percentage_deviation(selected_subsets_RE)

        pd.append(percentage_deviation)
        pd_RE.append(percentage_deviation_RE)

    print('CH1 Average Percentage Deviation:  ' + str(sum(pd) / len(pd)) + ' %')
    print('CH1 + RE Average Percentage Deviation: ' + str(sum(pd_RE) / len(pd_RE)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_RE[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from redundancy elimination using CH1')

    print('time: ' + str(time.time() - start) + '\n\n')

    # CH2
    start = time.time()

    pd = []
    pd_RE = []

    for instance in instance_objs:
        selected_subsets = CH2(instance)
        selected_subsets_RE = remove_redundant(instance, selected_subsets)

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)
        percentage_deviation_RE = instance.calculate_percentage_deviation(selected_subsets_RE)

        pd.append(percentage_deviation)
        pd_RE.append(percentage_deviation_RE)

    print('CH2 Average Percentage Deviation:  ' + str(sum(pd) / len(pd)) + ' %')
    print('CH2 + RE Average Percentage Deviation: ' + str(sum(pd_RE) / len(pd_RE)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_RE[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from redundancy elimination using CH2')

    print('time: ' + str(time.time() - start) + '\n\n')

    # CH3
    start = time.time()

    pd = []
    pd_RE = []

    for instance in instance_objs:
        selected_subsets = CH3(instance)
        selected_subsets_RE = remove_redundant(instance, selected_subsets)

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)
        percentage_deviation_RE = instance.calculate_percentage_deviation(selected_subsets_RE)

        pd.append(percentage_deviation)
        pd_RE.append(percentage_deviation_RE)

    print('CH3 Average Percentage Deviation:  ' + str(sum(pd) / len(pd)) + ' %')
    print('CH3 + RE Average Percentage Deviation: ' + str(sum(pd_RE) / len(pd_RE)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_RE[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from redundancy elimination using CH3')

    print('time: ' + str(time.time() - start) + '\n\n')

    # Only RE
    start = time.time()

    pd = []

    for instance in instance_objs:
        selected_subsets = remove_redundant(instance, [i for i in range(len(instance.subsets))])

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)

        pd.append(percentage_deviation)

    print('Only RE Average Percentage Deviation:  ' + str(sum(pd) / len(pd)) + ' %')

    print('time: ' + str(time.time() - start) + '\n\n\n\n')


def assignment2(instance_objs):
    print('Assignment 2\n')

    statistic_data = [[] for _ in range(len(instance_objs))]

    ################################
    #           CH1 + FI           #
    ################################

    start = time.time()

    pd = []
    pd_IH = []

    counter = 0

    for instance in instance_objs:
        selected_subsets = CH1(instance)
        selected_subsets_IH = improvement_heuristic(instance, selected_subsets, first_improvement=True)

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)
        percentage_deviation_IH = instance.calculate_percentage_deviation(selected_subsets_IH)

        pd.append(percentage_deviation)
        pd_IH.append(percentage_deviation_IH)
        statistic_data[counter].append(instance.calculate_cost(selected_subsets_IH))

        counter += 1
        # print(counter)

    print('CH1 Average Percentage Deviation: ' + str(sum(pd) / len(pd)) + ' %')
    print('CH1 + FI Average Percentage Deviation:  ' + str(sum(pd_IH) / len(pd_IH)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_IH[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from local search using CH1 + FI')

    print('time: ' + str(time.time() - start) + '\n\n')

    ################################
    #           CH2 + FI           #
    ################################

    start = time.time()

    pd = []
    pd_IH = []

    counter = 0

    for instance in instance_objs:
        selected_subsets = CH2(instance)
        selected_subsets_IH = improvement_heuristic(instance, selected_subsets, first_improvement=True)

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)
        percentage_deviation_IH = instance.calculate_percentage_deviation(selected_subsets_IH)

        pd.append(percentage_deviation)
        pd_IH.append(percentage_deviation_IH)
        statistic_data[counter].append(instance.calculate_cost(selected_subsets_IH))

        counter += 1
        # print(counter)

    print('CH2 Average Percentage Deviation: ' + str(sum(pd) / len(pd)) + ' %')
    print('CH2 + FI Average Percentage Deviation:  ' + str(sum(pd_IH) / len(pd_IH)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_IH[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from local search using CH2 + FI')

    print('time: ' + str(time.time() - start) + '\n\n')

    ################################
    #           CH3 + FI           #
    ################################

    start = time.time()

    pd = []
    pd_IH = []

    counter = 0

    for instance in instance_objs:
        selected_subsets = CH3(instance)
        selected_subsets_IH = improvement_heuristic(instance, selected_subsets, first_improvement=True)

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)
        percentage_deviation_IH = instance.calculate_percentage_deviation(selected_subsets_IH)

        pd.append(percentage_deviation)
        pd_IH.append(percentage_deviation_IH)
        statistic_data[counter].append(instance.calculate_cost(selected_subsets_IH))

        counter += 1
        # print(counter)

    print('CH3 Average Percentage Deviation: ' + str(sum(pd) / len(pd)) + ' %')
    print('CH3 + FI Average Percentage Deviation:  ' + str(sum(pd_IH) / len(pd_IH)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_IH[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from local search using CH3 + FI')

    print('time: ' + str(time.time() - start) + '\n\n')

    #####################################
    #           CH1 + RE + FI           #
    #####################################

    start = time.time()

    pd = []
    pd_IH = []

    counter = 0

    for instance in instance_objs:
        selected_subsets = CH1(instance)
        selected_subsets = remove_redundant(instance, selected_subsets)
        selected_subsets_IH = improvement_heuristic(instance, selected_subsets, first_improvement=True)

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)
        percentage_deviation_IH = instance.calculate_percentage_deviation(selected_subsets_IH)

        pd.append(percentage_deviation)
        pd_IH.append(percentage_deviation_IH)
        statistic_data[counter].append(instance.calculate_cost(selected_subsets_IH))

        counter += 1
        # print(counter)

    print('CH1 + RE Average Percentage Deviation: ' + str(sum(pd) / len(pd)) + ' %')
    print('CH1 + RE + FI Average Percentage Deviation:  ' + str(sum(pd_IH) / len(pd_IH)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_IH[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from local search using CH1 + RE + FI')

    print('time: ' + str(time.time() - start) + '\n\n')

    ################################
    #           CH1 + BI           #
    ################################

    start = time.time()

    pd = []
    pd_IH = []

    counter = 0

    for instance in instance_objs:
        selected_subsets = CH1(instance)
        selected_subsets_IH = improvement_heuristic(instance, selected_subsets, first_improvement=False)

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)
        percentage_deviation_IH = instance.calculate_percentage_deviation(selected_subsets_IH)

        pd.append(percentage_deviation)
        pd_IH.append(percentage_deviation_IH)
        statistic_data[counter].append(instance.calculate_cost(selected_subsets_IH))

        counter += 1
        # print(counter)

    print('CH1 Average Percentage Deviation: ' + str(sum(pd) / len(pd)) + ' %')
    print('CH1 + BI Average Percentage Deviation:  ' + str(sum(pd_IH) / len(pd_IH)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_IH[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from local search using CH1 + BI')

    print('time: ' + str(time.time() - start) + '\n\n')

    ################################
    #           CH2 + BI           #
    ################################

    start = time.time()

    pd = []
    pd_IH = []

    counter = 0

    for instance in instance_objs:
        selected_subsets = CH2(instance)
        selected_subsets_IH = improvement_heuristic(instance, selected_subsets, first_improvement=False)

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)
        percentage_deviation_IH = instance.calculate_percentage_deviation(selected_subsets_IH)

        pd.append(percentage_deviation)
        pd_IH.append(percentage_deviation_IH)
        statistic_data[counter].append(instance.calculate_cost(selected_subsets_IH))

        counter += 1
        # print(counter)

    print('CH2 Average Percentage Deviation: ' + str(sum(pd) / len(pd)) + ' %')
    print('CH2 + BI Average Percentage Deviation:  ' + str(sum(pd_IH) / len(pd_IH)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_IH[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from local search using CH2 + BI')

    print('time: ' + str(time.time() - start) + '\n\n')

    ################################
    #           CH3 + BI           #
    ################################

    start = time.time()

    pd = []
    pd_IH = []

    counter = 0

    for instance in instance_objs:
        selected_subsets = CH3(instance)
        selected_subsets_IH = improvement_heuristic(instance, selected_subsets, first_improvement=False)

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)
        percentage_deviation_IH = instance.calculate_percentage_deviation(selected_subsets_IH)

        pd.append(percentage_deviation)
        pd_IH.append(percentage_deviation_IH)
        statistic_data[counter].append(instance.calculate_cost(selected_subsets_IH))

        counter += 1
        # print(counter)

    print('CH3 Average Percentage Deviation: ' + str(sum(pd) / len(pd)) + ' %')
    print('CH3 + BI Average Percentage Deviation:  ' + str(sum(pd_IH) / len(pd_IH)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_IH[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from local search using CH3 + BI')

    print('time: ' + str(time.time() - start) + '\n\n')

    #####################################
    #           CH1 + RE + BI           #
    #####################################

    start = time.time()

    pd = []
    pd_IH = []

    counter = 0

    for instance in instance_objs:
        selected_subsets = CH1(instance)
        selected_subsets = remove_redundant(instance, selected_subsets)
        selected_subsets_IH = improvement_heuristic(instance, selected_subsets, first_improvement=False)

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)
        percentage_deviation_IH = instance.calculate_percentage_deviation(selected_subsets_IH)

        pd.append(percentage_deviation)
        pd_IH.append(percentage_deviation_IH)
        statistic_data[counter].append(instance.calculate_cost(selected_subsets_IH))

        counter += 1
        # print(counter)

    print('CH1 + RE Average Percentage Deviation: ' + str(sum(pd) / len(pd)) + ' %')
    print('CH1 + RE + BI Average Percentage Deviation:  ' + str(sum(pd_IH) / len(pd_IH)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_IH[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from local search using CH1 + RE + BI')

    print('time: ' + str(time.time() - start) + '\n\n')

    print('Statistic Data: \n')
    
    for i in statistic_data:
        for i2 in i:
            print(str(i2) + ' ', end='')
        print('')


def test_neighbours(instance):
    sol = CH1(instance)
    sol = remove_redundant(instance, sol)
    cost = instance.calculate_cost(sol)

    better = 0
    equal = 0
    equal_but_diff = 0
    worse = 0

    for new_sol, new_sol_cost, _ in next_neighbour(instance, sol):

        if new_sol_cost < cost:
            better += 1
        elif new_sol_cost == cost:
            equal += 1

            if set(new_sol) != set(sol):
                equal_but_diff += 1
        else:
            worse += 1

    print('better: ' + str(better))
    print('equal: ' + str(equal))
    print('equal_but_diff: ' + str(equal_but_diff))
    print('worse: ' + str(worse))
    print('total: ' + str(better+equal+worse))
    print()


def test_random_neighbours(instance):
    sol = CH1(instance)
    sol = remove_redundant(instance, sol)
    cost = instance.calculate_cost(sol)

    better = 0
    equal = 0
    equal_but_diff = 0
    worse = 0

    max_diff = 0

    for i, (new_sol, new_sol_cost, _) in enumerate(next_random_neighbour(instance, sol, n_subsets=50)):

        if new_sol_cost < cost:
            better += 1
        elif new_sol_cost == cost:
            equal += 1

            if set(new_sol) != set(sol):
                equal_but_diff += 1
        else:
            worse += 1

        if new_sol_cost - cost > max_diff:
            max_diff = new_sol_cost - cost

        if i == len(instance.subsets) * 10 - 1:
            break

    print('better: ' + str(better))
    print('equal: ' + str(equal))
    print('equal_but_diff: ' + str(equal_but_diff))
    print('worse: ' + str(worse))
    print('total: ' + str(better + equal + worse))
    print('max_diff: ' + str(max_diff))
    print()


def main():
    instances = [['SCP-Instances/scp42.txt', 512],
                 ['SCP-Instances/scp43.txt', 516],
                 ['SCP-Instances/scp44.txt', 494],
                 ['SCP-Instances/scp45.txt', 512],
                 ['SCP-Instances/scp46.txt', 560],
                 ['SCP-Instances/scp47.txt', 430],
                 ['SCP-Instances/scp48.txt', 492],
                 ['SCP-Instances/scp49.txt', 641],
                 ['SCP-Instances/scp51.txt', 253],
                 ['SCP-Instances/scp52.txt', 302],
                 ['SCP-Instances/scp53.txt', 226],
                 ['SCP-Instances/scp54.txt', 242],
                 ['SCP-Instances/scp55.txt', 211],
                 ['SCP-Instances/scp56.txt', 213],
                 ['SCP-Instances/scp57.txt', 293],
                 ['SCP-Instances/scp58.txt', 288],
                 ['SCP-Instances/scp59.txt', 279],
                 ['SCP-Instances/scp61.txt', 138],
                 ['SCP-Instances/scp62.txt', 146],
                 ['SCP-Instances/scp63.txt', 145],
                 ['SCP-Instances/scp64.txt', 131],
                 ['SCP-Instances/scp65.txt', 161],
                 ['SCP-Instances/scpa1.txt', 253],
                 ['SCP-Instances/scpa2.txt', 252],
                 ['SCP-Instances/scpa3.txt', 232],
                 ['SCP-Instances/scpa4.txt', 234],
                 ['SCP-Instances/scpa5.txt', 236],
                 ['SCP-Instances/scpb1.txt', 69],
                 ['SCP-Instances/scpb2.txt', 76],
                 ['SCP-Instances/scpb3.txt', 80],
                 ['SCP-Instances/scpb4.txt', 79],
                 ['SCP-Instances/scpb5.txt', 72],
                 ['SCP-Instances/scpc1.txt', 227],
                 ['SCP-Instances/scpc2.txt', 219],
                 ['SCP-Instances/scpc3.txt', 243],
                 ['SCP-Instances/scpc4.txt', 219],
                 ['SCP-Instances/scpc5.txt', 215],
                 ['SCP-Instances/scpd1.txt', 60],
                 ['SCP-Instances/scpd2.txt', 66],
                 ['SCP-Instances/scpd3.txt', 72],
                 ['SCP-Instances/scpd4.txt', 62],
                 ['SCP-Instances/scpd5.txt', 61]]

    instance_objs = []

    for i in instances:
        instance_objs.append(Instance(i[0], i[1]))

    # assignment1(instance_objs)
    # assignment2(instance_objs)

    # for instance in instance_objs:
        # test_neighbours(instance)
        # test_random_neighbours(instance)

    '''index = 0

    sol = CH1(instance_objs[index])
    sol = remove_redundant(instance_objs[index], sol)

    sol = simulated_annealing(instance_objs[index], sol)'''

    grasp(instance_objs[0], 500, 0.05)


main()

'''
150

temperature = -150/ln(0.5) = 216

2*temperature = 216*2 = 432
'''


''''# plt.axis([-50,50,0,10000])
plt.ion()
plt.show()

x = np.arange(-50, 51)
for pow in range(1,5):   # plot x^1, x^2, ..., x^4
    y = [Xi**pow for Xi in x]
    print(plt.plot(x, y))
    plt.draw()
    plt.pause(1)
    # input("Press [enter] to continue.")'''

