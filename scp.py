# Leonardo Gomes Capozzi - up201503708

import random
import time

seed = 10


def read_file(filename):

    file = open(filename, 'r')

    size = file.readline().split()

    n_universe = int(size[0])
    n_subsets = int(size[1])

    universe = [i for i in range(n_universe)]
    subset_weights = []
    subsets = [[] for _ in range(n_subsets)]
    elem_subsets = []

    while len(subset_weights) < n_subsets:
        line = file.readline().split()

        for e in line:
            subset_weights.append(int(e))

    curr_elem = 0

    while curr_elem < n_universe:
        n_elems = int(file.readline().split()[0])

        elem_subset = []

        while n_elems > 0:
            line = file.readline().split()

            for e in line:
                subsets[int(e)-1].append(curr_elem)
                elem_subset.append(int(e)-1)
                n_elems -= 1

        elem_subsets.append(elem_subset)

        curr_elem += 1

    file.close()

    return universe, subset_weights, subsets, elem_subsets


def calculate_cost(selected_subsets, subset_weights):
    cost = 0

    for e in selected_subsets:
        cost += subset_weights[e]

    return cost


# random
def method1(universe, subset_weights, subsets, elem_subsets):
    subsets_aux = []

    for s in subsets:
        subsets_aux.append(set(s))

    subsets = subsets_aux
    universe = set(universe)
    selected_subsets = []
    selected_elements = set()
    remaining_elems = universe

    while selected_elements != universe:
        remaining_elems = remaining_elems.difference(selected_elements)

        random.seed(seed)
        rand_elem = random.sample(remaining_elems, 1)[0]

        random.seed(seed)
        rand_subset = random.choice(elem_subsets[rand_elem])

        selected_subsets.append(rand_subset)

        selected_elements = selected_elements.union(subsets[rand_subset])

    return selected_subsets


# random elem, choose subset with min cost
def method2(universe, subset_weights, subsets, elem_subsets):
    subsets_aux = []

    for s in subsets:
        subsets_aux.append(set(s))

    subsets = subsets_aux

    universe = set(universe)
    selected_subsets = []
    selected_elements = set()
    remaining_elems = universe

    while selected_elements != universe:
        remaining_elems = remaining_elems.difference(selected_elements)

        random.seed(seed)
        rand_elem = random.sample(remaining_elems, 1)[0]

        subset = min(elem_subsets[rand_elem], key=lambda s: subset_weights[s])

        selected_subsets.append(subset)

        selected_elements = selected_elements.union(subsets[subset])

    return selected_subsets


# choose subset with min heuristic value
def method3(universe, subset_weights, subsets, elem_subsets):
    subsets_aux = []

    for s in subsets:
        subsets_aux.append(set(s))

    subsets = subsets_aux

    universe = set(universe)
    selected_subsets = []
    selected_elements = set()
    remaining_elems = universe

    while selected_elements != universe:
        remaining_elems = remaining_elems.difference(selected_elements)

        aval_subsets = []

        for e in remaining_elems:
            aval_subsets.extend(elem_subsets[e])

        aval_subsets = list(set(aval_subsets))

        subset = min(aval_subsets, key=lambda s: subset_weights[s]/len(subsets[s].difference(selected_elements)))

        selected_subsets.append(subset)

        selected_elements = selected_elements.union(subsets[subset])

    return selected_subsets


def remove_redundant(universe, subset_weights, subsets, elem_subsets, selected_subsets):
    selected_subsets = selected_subsets.copy()

    universe = set(universe)

    selected_subsets.sort(reverse=True, key=lambda e: subset_weights[e])

    i = 0

    while i < len(selected_subsets):
        removed_subset = selected_subsets.pop(i)

        selected_elements = []

        for s in selected_subsets:
            selected_elements.extend(subsets[s])

        selected_elements = set(selected_elements)

        if selected_elements != universe:
            selected_subsets.insert(i, removed_subset)
            i += 1

    return selected_subsets


def calculate_percentage_deviation_CH1(file, best_solution, remove_redundancy=False):
    universe, subset_weights, subsets, elem_subsets = read_file(file)
    selected_subsets = method1(universe, subset_weights, subsets, elem_subsets)

    if remove_redundancy:
        selected_subsets = remove_redundant(universe, subset_weights, subsets, elem_subsets, selected_subsets)

    cost = calculate_cost(selected_subsets, subset_weights)

    pd = (abs(cost - best_solution)*100)/best_solution

    return pd


def calculate_percentage_deviation_CH2(file, best_solution, remove_redundancy=False):
    universe, subset_weights, subsets, elem_subsets = read_file(file)
    selected_subsets = method2(universe, subset_weights, subsets, elem_subsets)

    if remove_redundancy:
        selected_subsets = remove_redundant(universe, subset_weights, subsets, elem_subsets, selected_subsets)

    cost = calculate_cost(selected_subsets, subset_weights)

    pd = (abs(cost - best_solution)*100)/best_solution

    return pd


def calculate_percentage_deviation_CH3(file, best_solution, remove_redundancy=False):
    universe, subset_weights, subsets, elem_subsets = read_file(file)
    selected_subsets = method3(universe, subset_weights, subsets, elem_subsets)

    if remove_redundancy:
        selected_subsets = remove_redundant(universe, subset_weights, subsets, elem_subsets, selected_subsets)

    cost = calculate_cost(selected_subsets, subset_weights)

    pd = (abs(cost - best_solution)*100)/best_solution

    return pd


def calculate_percentage_deviation_CH4(file, best_solution):
    universe, subset_weights, subsets, elem_subsets = read_file(file)
    selected_subsets = remove_redundant(universe, subset_weights, subsets, elem_subsets, [i for i in range(len(subsets))])

    cost = calculate_cost(selected_subsets, subset_weights)

    pd = (abs(cost - best_solution) * 100) / best_solution

    return pd


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

    # CH1 no redundancy elimination
    start = time.time()

    pd = []

    for i in instances:
        pd.append(calculate_percentage_deviation_CH1(i[0], i[1]))

    print('CH1 Average Percentage Deviation:  ' + str(sum(pd)/len(pd)) + ' %')

    # CH1 redundancy elimination
    pd2 = []

    for i in instances:
        pd2.append(calculate_percentage_deviation_CH1(i[0], i[1], remove_redundancy=True))

    print('CH1R Average Percentage Deviation: ' + str(sum(pd2)/len(pd2)) + ' %')

    # CH1 check redundancy profit

    profits = 0

    for i in range(len(instances)):
        if pd2[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instances)) + ' instances profit from redundancy elimination using CH1\n\n')

    print('time: ' + str(time.time()-start))

    # CH2 no redundancy elimination
    start = time.time()

    pd = []

    for i in instances:
        pd.append(calculate_percentage_deviation_CH2(i[0], i[1]))

    print('CH2 Average Percentage Deviation:  ' + str(sum(pd) / len(pd)) + ' %')

    # CH2 redundancy elimination
    pd2 = []

    for i in instances:
        pd2.append(calculate_percentage_deviation_CH2(i[0], i[1], remove_redundancy=True))

    print('CH2R Average Percentage Deviation: ' + str(sum(pd2) / len(pd2)) + ' %')

    # CH2 check redundancy profit

    profits = 0

    for i in range(len(instances)):
        if pd2[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instances)) + ' instances profit from redundancy elimination using CH2\n\n')

    print('time: ' + str(time.time() - start))

    # CH3 no redundancy elimination
    start = time.time()

    pd = []

    for i in instances:
        pd.append(calculate_percentage_deviation_CH3(i[0], i[1]))

    print('CH3 Average Percentage Deviation:  ' + str(sum(pd) / len(pd)) + ' %')

    # CH3 redundancy elimination
    pd2 = []

    for i in instances:
        pd2.append(calculate_percentage_deviation_CH3(i[0], i[1], remove_redundancy=True))

    print('CH3R Average Percentage Deviation: ' + str(sum(pd2) / len(pd2)) + ' %')

    # CH3 check redundancy profit

    profits = 0

    for i in range(len(instances)):
        if pd2[i] < pd[i]:
            profits += 1

    print(str(profits) + '/' + str(len(instances)) + ' instances profit from redundancy elimination using CH3\n\n')

    print('time: ' + str(time.time() - start))

    # CH4
    start = time.time()

    pd = []

    for i in instances:
        pd.append(calculate_percentage_deviation_CH4(i[0], i[1]))

    print('CH4 Average Percentage Deviation:  ' + str(sum(pd) / len(pd)) + ' %')

    print('time: ' + str(time.time() - start))


main()

