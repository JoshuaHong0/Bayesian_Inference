from exact_inferencer import normalize
import random
import BayesianNetwork
import sys


def likelihood_weighting(x, evidence, bn, n):
    """
    Input:
        x, the query variable
        e, observed values for variables E
        bn, a Bayesian network specifying joint distribution P(X1,...,Xn)
        n, the total number of samples to be generated
    Output:
        an estimate of P(X|e)
    """
    count = [0, 0]
    for i in range(n):
        sample = weighted_sample(bn, evidence)
        w = sample["weight"]
        index = 0 if sample[x] else 1
        count[index] += w
    return normalize(count)


def weighted_sample(bn, e):
    """
    Input:
        bn, a Bayesian network
        e, evidence
    Output:
        x, an event
        w, a weight
    """
    w = 1.0
    sampled = {}
    for key in e:
        sampled[key] = e[key]
    node_list = bn.get_nodelist()
    for node in node_list:
        parent_val = []
        node_parent = node.get_parent()
        node_name = node.get_name()
        for parent in node_parent:
            parent_val.append(sampled[parent])
        if node_name in e:
            w = w * node.get_probability(e[node_name], parent_val)
        else:
            p_distribution = [node.get_probability(True, parent_val), node.get_probability(False, parent_val)]
            val = random.choices([True, False], p_distribution)
            sampled[node_name] = bool(val[0])
    sampled["weight"] = w
    return sampled


def main():
    """
    Likelihood weighting main function
    """
    n = int(sys.argv[1])
    file_name = sys.argv[2]
    x = sys.argv[3]
    bn = BayesianNetwork.xml_reader(file_name, "BayesianNet")
    e = {}
    for index in range(4, len(sys.argv), 2):
        e[sys.argv[index]] = bool(sys.argv[index+1])
        index += 2
    print(likelihood_weighting(x, e, bn, n))


if __name__ == '__main__':
    main()


