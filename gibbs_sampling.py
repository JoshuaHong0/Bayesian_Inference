import BayesianNetwork
import random
from exact_inferencer import normalize
import sys


def gibbs_ask(x, e, bn, n):
    """
    Input:
        x, query variable
        e, given evidence
        bn, Bayesian net
        n, the total number of samples to be generated
    Output:
        an estimate of P(X|e)
    """
    count = [0, 0]
    node_list = bn.get_nodelist()
    current_state = {}
    z = []
    """
    Initialization
    """
    for node in node_list:
        node_name = node.get_name()
        if node_name in e:
            current_state[node_name] = e[node_name]
        else:
            z.append(node_name)
            random_val = random.choice([True, False])
            current_state[node_name] = random_val

    for time in range(n):
        for val in z:
            val_node = bn.get_node(val)
            val_parent = val_node.get_parent()
            p_distribution = []
            for parent in val_parent:
                p_distribution.append(current_state[parent])
            p_true = val_node.get_probability(True, p_distribution)
            p_false = val_node.get_probability(False, p_distribution)
            val_children = val_node.get_children()
            for child in val_children:
                child_node = bn.get_node(child)
                child_parent = child_node.get_parent()
                
                current_state[val] = True
                cp_distribution = []
                for parent in child_parent:
                    cp_distribution.append(current_state[parent])
                p_true *= child_node.get_probability(current_state[child], cp_distribution)

                current_state[val] = False
                cp_distribution.clear()
                for parent in child_parent:
                    cp_distribution.append(current_state[parent])
                p_false *= child_node.get_probability(current_state[child], cp_distribution)

            new = random.choices([True, False], [p_true, p_false])
            current_state[val] = bool(new[0])
            index = 0 if current_state[x] else 1
            count[index] += 1

    return normalize(count)


def main():
    """
    Gibbs sampling main function
    """
    n = int(sys.argv[1])
    file_name = sys.argv[2]
    x = sys.argv[3]
    bn = BayesianNetwork.xml_reader(file_name, "BayesianNet")
    e = {}
    for index in range(4, len(sys.argv), 2):
        e[sys.argv[index]] = bool(sys.argv[index+1])
        index += 2
    print(gibbs_ask(x, e, bn, n))


if __name__ == '__main__':
    main()


