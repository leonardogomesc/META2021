# Leonardo Gomes Capozzi - up201503708

import random
import time
import numpy as np

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


def remove_redundant(instance, selected_subsets, element_count=None, return_cost=False):
    # copy selected subsets
    selected_subsets = selected_subsets.copy()

    # sort by cost per element
    selected_subsets.sort(key=lambda e: instance.subset_weights[e]/len(instance.subsets[e]))

    # calculate or copy array containing the number of times each element has been selected
    if element_count is None:
        element_count = np.zeros(instance.matrix.shape[1])

        for s in selected_subsets:
            element_count = element_count + instance.matrix[s]
    else:
        element_count = element_count.copy()

    # calculate cost
    cost = 0

    i = len(selected_subsets) - 1

    while i >= 0:
        # try to remove subset
        element_count = element_count - instance.matrix[selected_subsets[i]]

        # check if there is an uncovered element
        if 0 not in element_count:
            # remove subset if there is not an uncovered element
            selected_subsets.pop(i)
        else:
            # add subset if there is an uncovered element
            element_count = element_count + instance.matrix[selected_subsets[i]]
            cost += instance.subset_weights[selected_subsets[i]]

        i -= 1

    if return_cost:
        return selected_subsets, cost

    return selected_subsets


# choose subset with min heuristic value
def CH1(instance):
    selected_subsets = []  # index of selected subsets
    covered_elements = set()  # covered elements
    uncovered_elements = instance.universe_set  # uncovered elements

    while covered_elements != instance.universe_set:
        # calculate uncovered elements
        uncovered_elements = uncovered_elements.difference(covered_elements)

        # calculate subsets with at least one uncovered element
        aval_subsets = []

        for e in uncovered_elements:
            aval_subsets.extend(instance.elem_subsets[e])

        aval_subsets = list(set(aval_subsets))

        # choose subset with min cost
        subset = min(aval_subsets, key=lambda s: instance.subset_weights[s] / len(instance.subsets_set[s].difference(covered_elements)))

        # add subset to list
        selected_subsets.append(subset)

        # update covered elements
        covered_elements = covered_elements.union(instance.subsets_set[subset])

    return selected_subsets


# choose element with min num of subsets containing it then subset with min heuristic value
def CH2(instance):
    selected_subsets = []  # index of selected subsets
    covered_elements = set()  # covered elements
    uncovered_elements = instance.universe_set  # uncovered elements

    while covered_elements != instance.universe_set:
        # calculate uncovered elements
        uncovered_elements = uncovered_elements.difference(covered_elements)

        # calculate uncovered element with lowest amount of subsets containing it
        elem = min(uncovered_elements, key=lambda e: len(instance.elem_subsets[e]))

        # choose subset containing elem with the lowest cost
        subset = min(instance.elem_subsets[elem], key=lambda s: instance.subset_weights[s])

        # add subset to list
        selected_subsets.append(subset)

        # update covered elements
        covered_elements = covered_elements.union(instance.subsets_set[subset])

    return selected_subsets


# random elem, choose subset with min cost
def CH3(instance):
    selected_subsets = []  # index of selected subsets
    covered_elements = set()  # covered elements
    uncovered_elements = instance.universe_set  # uncovered elements

    while covered_elements != instance.universe_set:
        # calculate uncovered elements
        uncovered_elements = uncovered_elements.difference(covered_elements)

        # select random uncovered element
        random.seed(seed)
        rand_elem = random.sample(uncovered_elements, 1)[0]

        # choose subset containing elem with the lowest cost
        subset = min(instance.elem_subsets[rand_elem], key=lambda s: instance.subset_weights[s])

        # add subset to list
        selected_subsets.append(subset)

        # update covered elements
        covered_elements = covered_elements.union(instance.subsets_set[subset])

    return selected_subsets


def next_neighbour(instance, curr_sol):
    # copy solution list
    curr_sol = curr_sol.copy()

    # calculate list of subsets that are not in the current solution
    remaining_subsets = list(instance.subset_universe_set.difference(set(curr_sol)))

    # sort by cost per element
    remaining_subsets.sort(key=lambda e: instance.subset_weights[e]/len(instance.subsets[e]))

    # calculate array containing the number of times each element has been selected
    element_count = np.zeros(instance.matrix.shape[1])

    for s in curr_sol:
        element_count = element_count + instance.matrix[s]

    for rs in remaining_subsets:
        # append a subset to the current solution
        curr_sol.append(rs)
        element_count = element_count + instance.matrix[rs]

        # remove redundancy
        yield remove_redundant(instance, curr_sol, element_count, return_cost=True)

        # remove the subset that was added
        curr_sol.pop()
        element_count = element_count - instance.matrix[rs]


def improvement_heuristic(instance, selected_subsets, first_improvement=True):
    curr_sol = selected_subsets.copy()  # current solution
    curr_sol_cost = instance.calculate_cost(curr_sol)  # current solution cost

    failed = False

    while not failed:
        failed = True

        # create neighbour generator
        iter_neighbour = next_neighbour(instance, curr_sol)

        for new_sol, new_sol_cost in iter_neighbour:

            # check if new solution has a lower cost than current solution
            if new_sol_cost < curr_sol_cost:
                curr_sol = new_sol
                curr_sol_cost = new_sol_cost
                failed = False

                # if first improvement take a step
                if first_improvement:
                    break

    return curr_sol


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

    assignment1(instance_objs)
    assignment2(instance_objs)


main()

