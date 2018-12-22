import BayesianNetwork
from exact_inferencer import normalize
import sys


def point_wise_product(f1, f2):
    """
    Point-wise product function
    Input:
        var, Selected hidden variable
        f1,f2, Two factors
    Output:
        the point-wise product of f1 and f2
    """
    res = BayesianNetwork.Factor()
    hidden = set(f1.hidden_val + f2.hidden_val)
    permutation = BayesianNetwork.generate_permutation(len(hidden))
    table = []
    for i in range(len(permutation)):
        record = {}
        tmp_line = []
        index = 0
        prob = 1.0
        for val in hidden:
            record[val] = permutation[i][index]
            index += 1
        table1 = f1.prob_table
        table2 = f2.prob_table
        for line in table1:
            if consist(record, line[0]):
                prob *= line[1]
        for line in table2:
            if consist(record, line[0]):
                prob *= line[1]
        tmp_line.append(record)
        tmp_line.append(prob)
        table.append(tmp_line)
    res.prob_table = table
    res.hidden_val = list(hidden)
    return res


def sum_out(var, f):
    """
    Sum out function
    Input:
        var, the variable to be summed out
        f, a factor (the point-wise product of all factors that have var as a hidden variable)
    Output:
        The summed out result of f
    """
    res = BayesianNetwork.Factor()
    new_hidden_vals = []
    for val in f.hidden_val:
        if val != var:
            new_hidden_vals.append(val)
    permutation = BayesianNetwork.generate_permutation(len(new_hidden_vals))
    for i in range(len(permutation)):
        record = {}
        tmp_line = []
        index = 0
        prob = 0
        for val in new_hidden_vals:
            record[val] = permutation[i][index]
            index += 1
        for line in f.prob_table:
            if consist(line[0], record):
                prob += line[1]
        tmp_line.append(record)
        tmp_line.append(prob)
        res.prob_table.append(tmp_line)
        res.hidden_val = new_hidden_vals
    return res


def consist(dict1, dict2):
    """
    Input:
        Two dictionaries (The first dictionary has larger size)
    Output:
        Whether dict2 is consistent with dict1
    """
    for item in dict1:
        if item in dict2:
            if dict1[item] != dict2[item]:
                return False
    return True


def elimination_ask(x, e, bn):
    """
    Variable Elimination function
    Input:
        X, query variable
        e, observed values for variable E
        bn, a Bayesian network
    Output:
        a distribution over X
    """
    nodes = []
    node_list = bn.get_nodelist()
    factors = []
    hidden = []
    for node in node_list:
        node_name = node.get_name()
        nodes.append(node_name)
        factor = BayesianNetwork.make_factor(node, e)
        factors.append(factor)
    for node_name in nodes:
        if node_name not in e and node_name != x:
            hidden.append(node_name)
    index = -1
    new_factor = factors[index]
    if nodes[index] in hidden:
        new_factor = sum_out(nodes[index], new_factor)
    for index in range(-2, -len(nodes)-1, -1):
        new_factor = point_wise_product(factors[index], new_factor)
        if nodes[index] in hidden:
            new_factor = sum_out(nodes[index], new_factor)
    count = [0, 0]
    for line in new_factor.prob_table:
        if line[0][x]:
            count[0] = line[1]
        else:
            count[1] = line[1]
    return normalize(count)


def main():
    """
    Variable elimination main function
    """
    file_name = sys.argv[1]
    bn = BayesianNetwork.xml_reader(file_name, "BayesianNet")
    x = sys.argv[2]
    e = {}
    for index in range(3, len(sys.argv), 2):
        e[sys.argv[index]] = bool(sys.argv[index+1])
        index += 2
    print(elimination_ask(x, e, bn))


if __name__ == '__main__':
    main()