import BayesianNetwork
import sys


def enumerate_ask(x, e, bn):
    """
    Input:
        x, the query variable
        e, observed values for variables E
        bn, a Bayes net with variables {X} ∪ E ∪ Y
    Output:
        a distribution over X
    """
    q = []
    bn_vars = []
    node_list = bn.get_nodelist()
    for node in node_list:
        bn_vars.append(node.get_name())
    e[x] = True
    q.append(enumerate_all(bn_vars, e, bn))
    e[x] = False
    q.append(enumerate_all(bn_vars, e, bn))
    return normalize(q)


def enumerate_all(bn_vars, e, bn):
    """
    Enumerate all possible assignment
    Input:
        bn_vars, a list of all variables in Bayes net
        e, observed values
    Output:
        a real number
    """
    if not bn_vars:
        return 1.0
    y = bn_vars[0]
    tmp = bn_vars.copy()
    tmp.pop(0)
    node = bn.get_node(y)
    parent = node.get_parent()
    p = []
    for item in parent:
        p.append(e[item])
    if y in e:
        e_copy = e.copy()
        return node.get_probability(e[y], p) * enumerate_all(tmp, e_copy, bn)
    else:
        e_copy = e.copy()
        e_copy[y] = True
        first = node.get_probability(True, p) * enumerate_all(tmp, e_copy, bn)
        e_copy[y] = False
        second = node.get_probability(False, p) * enumerate_all(tmp, e_copy, bn)
        return first + second


def normalize(distribution=[]):
    """
    Normalization function
    """
    res = []
    total = 0
    for item in distribution:
        total += item
    if total == 0:
        return "We don't have enough samples"
    for item in distribution:
        res.append(round(item/total, 3))
    return res


def main():
    """
    Exact inference main function
    """
    file_name = sys.argv[1]
    bn = BayesianNetwork.xml_reader(file_name, "BayesianNet")
    x = sys.argv[2]
    e = {}
    for index in range(3, len(sys.argv), 2):
        e[sys.argv[index]] = bool(sys.argv[index+1])
        index += 2
    print(enumerate_ask(x, e, bn))


if __name__ == '__main__':
    main()
