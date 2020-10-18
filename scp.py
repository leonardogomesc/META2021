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
        self.subset_weights = subset_weights
        self.subsets = subsets
        self.elem_subsets = elem_subsets
        self.best_solution = best_solution

        self.universe_set = set(self.universe)

        subsets_set = []

        for s in self.subsets:
            subsets_set.append(set(s))

        self.subsets_set = subsets_set

        matrix = np.zeros((len(self.subsets), len(self.elem_subsets)))

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


def remove_redundant(instance, selected_subsets):
    selected_subsets = selected_subsets.copy()

    selected_subsets.sort(reverse=True, key=lambda e: instance.subset_weights[e])

    i = 0

    while i < len(selected_subsets):
        removed_subset = selected_subsets.pop(i)

        selected_elements = []

        for s in selected_subsets:
            selected_elements.extend(instance.subsets[s])

        selected_elements = set(selected_elements)

        if selected_elements != instance.universe_set:
            selected_subsets.insert(i, removed_subset)
            i += 1

    return selected_subsets


# random
def CH1(instance):
    selected_subsets = []
    selected_elements = set()
    remaining_elems = instance.universe_set

    while selected_elements != instance.universe_set:
        remaining_elems = remaining_elems.difference(selected_elements)

        random.seed(seed)
        rand_elem = random.sample(remaining_elems, 1)[0]

        random.seed(seed)
        rand_subset = random.choice(instance.elem_subsets[rand_elem])

        selected_subsets.append(rand_subset)

        selected_elements = selected_elements.union(instance.subsets_set[rand_subset])

    return selected_subsets


# random elem, choose subset with min cost
def CH2(instance):
    selected_subsets = []
    selected_elements = set()
    remaining_elems = instance.universe_set

    while selected_elements != instance.universe_set:
        remaining_elems = remaining_elems.difference(selected_elements)

        random.seed(seed)
        rand_elem = random.sample(remaining_elems, 1)[0]

        subset = min(instance.elem_subsets[rand_elem], key=lambda s: instance.subset_weights[s])

        selected_subsets.append(subset)

        selected_elements = selected_elements.union(instance.subsets_set[subset])

    return selected_subsets


# choose subset with min heuristic value
def CH3(instance):
    selected_subsets = []
    selected_elements = set()
    remaining_elems = instance.universe_set

    while selected_elements != instance.universe_set:
        remaining_elems = remaining_elems.difference(selected_elements)

        aval_subsets = []

        for e in remaining_elems:
            aval_subsets.extend(instance.elem_subsets[e])

        aval_subsets = list(set(aval_subsets))

        subset = min(aval_subsets, key=lambda s: instance.subset_weights[s] / len(
            instance.subsets_set[s].difference(selected_elements)))

        selected_subsets.append(subset)

        selected_elements = selected_elements.union(instance.subsets_set[subset])

    return selected_subsets


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
    print('CH1 RE Average Percentage Deviation: ' + str(sum(pd_RE) / len(pd_RE)) + ' %')

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
    print('CH2 RE Average Percentage Deviation: ' + str(sum(pd_RE) / len(pd_RE)) + ' %')

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
    print('CH3 RE Average Percentage Deviation: ' + str(sum(pd_RE) / len(pd_RE)) + ' %')

    profits = 0

    for i in range(len(instance_objs)):
        if pd_RE[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instance_objs)) + ' instances profit from redundancy elimination using CH3')

    print('time: ' + str(time.time() - start) + '\n\n')

    # CH4
    start = time.time()

    pd = []

    for instance in instance_objs:
        selected_subsets = remove_redundant(instance, [i for i in range(len(instance.subsets))])

        percentage_deviation = instance.calculate_percentage_deviation(selected_subsets)

        pd.append(percentage_deviation)

    print('CH4 Average Percentage Deviation:  ' + str(sum(pd) / len(pd)) + ' %')

    print('time: ' + str(time.time() - start))


main()
