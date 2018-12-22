import BayesianNetwork
import random
from exact_inferencer import normalize
import sys


def prior_sample(bn):
    """
    Input:
        A Bayesian network
    Output:
        A randomly sampled event from the prior specified by input Bayes net
    """
    node_list = bn.get_nodelist()
    sampled = {}
    for node in node_list:
        parent_val = []
        node_parent = node.get_parent()
        for parent in node_parent:
            parent_val.append(sampled[parent])
        p_distribution = [node.get_probability(True, parent_val), node.get_probability(False, parent_val)]
        val = random.choices([True, False], p_distribution)
        sampled[node.get_name()] = bool(val[0])
    return sampled


def rejection_sampling(x, e, bn, n):
    """
    Input:
        x, the query variable
        e, observed values for variables E
        bn, a Bayesian network
        N, the total number of samples to be generated
    Output:
        estimate of probability x given evidence e
    """
    counts = [0, 0]
    for num in range(n):
        event = prior_sample(bn)
        if consistent(event, e):
            index = 0 if event[x] else 1
            counts[index] += 1
    return normalize(counts)


def consistent(x, e):
    """
    Input:
        Two dictionaries:
            1. A sampled event
            2. Given evidence
    Output:
        Whether this sampled event is consistent with given evidence
    """
    keys = x.keys()
    for key in keys:
        if key in e:
            if e[key] != x[key]:
                return False
    return True


def main():
    """
    Rejection sampling main function
    """
    n = int(sys.argv[1])
    file_name = sys.argv[2]
    x = sys.argv[3]
    bn = BayesianNetwork.xml_reader(file_name, "BayesianNet")
    e = {}
    for index in range(4, len(sys.argv), 2):
        e[sys.argv[index]] = bool(sys.argv[index+1])
        index += 2
    print(rejection_sampling(x, e, bn, n))


if __name__ == '__main__':
    main()
