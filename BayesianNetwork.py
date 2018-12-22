
import xml.dom.minidom
import itertools


class BayesianNet(object):
    """
    Class of Bayesian Network
    """
    def __init__(self, name):
        """
        Initialization
        """
        self.name = name
        self.nodes = []

    def add_node(self, node):
        """
        Add a node to the network
        """
        self.nodes.append(node)

    def get_node(self, target_name):
        """
        Input: a node name
        Output: the corresponding node object
        """
        for node in self.nodes:
            if node.name == target_name:
                return node

    def get_nodelist(self):
        """
        Get a list of all the node objects in this Bayesian network
        """
        return self.nodes


class Node(object):
    """
    Class of Node
    """
    def __init__(self, name):
        """
        Initialization
        """
        self.name = name
        self.parents = []
        self.children = []
        self.outcomes = []
        self.probabilities = []

    def add_outcome(self, name):
        """
        Add outcomes to the node
        """
        self.outcomes.append(name)

    def set_probabilities(self, p=[]):
        """
        Set the probabilities (conditional probability table) for the node
        """
        self.probabilities = p[:]

    def get_probability(self, node, parents=[]):
        """
        Get the conditional/prior probability of the node
        """
        if not parents:
            """
            Get prior probability
            """
            if node:
                return float(self.probabilities[0].split()[0])
            else:
                return float(self.probabilities[0].split()[1])
        else:
            """
            Get conditional probability
            """
            n = len(parents) - 1
            index = 0
            for p in parents:
                index += 2**n if p is False else 0
                n -= 1
            if node:
                return float(self.probabilities[index].split()[0])
            else:
                return float(self.probabilities[index].split()[1])

    def add_parent(self, parent):
        """
        Add parents to the node
        """
        self.parents.append(parent)

    def add_child(self, child):
        """
        Add child to the node
        """
        self.children.append(child)

    def get_name(self):
        """
        Get the name of the node
        """
        return self.name

    def get_probabilities(self):
        """
        Get a list representing the conditional probability table of the node
        """
        return self.probabilities

    def get_parent(self):
        """
        Get a list of parents of this node
        """
        return self.parents

    def get_children(self):
        """
        Get a list of children of this node
        """
        return self.children


def xml_reader(filename, name):
    """
    The XML parser: Read a XML file and fit it into a Bayesian Network
    """
    bn = BayesianNet(name)
    document = xml.dom.minidom.parse(filename)
    node_y = {}
    y = []
    node_x = {}
    x = []
    """
    Parse data with tag: VARIABLE
    """
    for variable in document.getElementsByTagName("VARIABLE"):
        node_name = variable.getElementsByTagName("NAME")[0].childNodes[0].nodeValue
        outcomes = variable.getElementsByTagName("OUTCOME")
        properties = variable.getElementsByTagName("PROPERTY")
        temp_node = Node(node_name)
        for outcome in outcomes:
            o = outcome.childNodes[0].nodeValue
            temp_node.add_outcome(o)
        if not properties:
            bn.add_node(temp_node)
        else:
            node_position = properties[0].childNodes[0].nodeValue
            pos_comma = node_position.index(',')
            open_p = node_position.index('(')
            height = int(node_position[pos_comma+1: -1])
            node_y[temp_node] = height
            y.append(height)
            width = int(node_position[open_p+1: pos_comma])
            node_x[temp_node] = width
            x.append(width)
    """
    Handle special case for dog-problem in which each variable has a coordinate
    """
    if y:
        set_y = set(y)
        sorted_list_y = sorted(set_y)
        set_x = set(x)
        sorted_list_x = sorted(set_x)
        for h in sorted_list_y:
            # node objects of the same height
            same_y_nodes = []
            for node_object in node_y:
                if node_y[node_object] == h:
                    same_y_nodes.append(node_object)
            for w in sorted_list_x:
                for node_object in same_y_nodes:
                    if node_x[node_object] == w:
                        bn.add_node(node_object)

    """
    Parse data with Tag DEFINITION
    """
    for definition in document.getElementsByTagName("DEFINITION"):
        node_name = definition.getElementsByTagName("FOR")[0].childNodes[0].nodeValue
        evidence = definition.getElementsByTagName("GIVEN")
        table = definition.getElementsByTagName("TABLE")[0]
        temp_node = bn.get_node(node_name)
        for e in evidence:
            e_name = e.childNodes[0].nodeValue
            temp_node.add_parent(e_name)
            parent_node = bn.get_node(e_name)
            parent_node.add_child(node_name)
        p = get_text(table.childNodes)
        if not y:
            temp_node.set_probabilities(list(filter(None, p)))
        else:
            p_elements = p[0].split()
            right_format = []
            for i in range(0, len(p_elements), 2):
                item = p_elements[i] + ' ' + p_elements[i+1]
                right_format.append(item)
            temp_node.set_probabilities(right_format)
    return bn


def get_text(nodelist):
    """
    Transform the data in nodelist into proper text form
    """
    res = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            res.append(node.data.strip(' \t\n\r'))
    return res


class Factor(object):
    """
    Class of factor
    """
    def __init__(self):
        self.node_name = ''
        self.prob_table = []
        self.parents = []
        self.hidden_val = []


def make_factor(node_object, e):
    """
    Initialize a factor
    Input:
        A node object,
        evidence
    Output:
        A factor
    """
    f = Factor()
    f.node_name = node_object.get_name()
    f.parents = node_object.get_parent()
    if f.node_name not in e:
        f.hidden_val.append(f.node_name)
    for parent in f.parents:
        if parent not in e:
            f.hidden_val.append(parent)
    cnt_e = len(f.hidden_val)
    permutation = generate_permutation(cnt_e)
    for i in range(len(permutation)):
        index = 0
        perm = []
        line = []
        x = f.node_name
        record = {}
        if x in e:
            perm.append(e[x])
            record[x] = e[x]
        else:
            perm.append(permutation[i][index])
            record[x] = permutation[i][index]
            index += 1
        for parent in f.parents:
            if parent in e:
                perm.append(e[parent])
                record[parent] = e[parent]
            else:
                perm.append(permutation[i][index])
                record[parent] = permutation[i][index]
                index += 1
        prob = node_object.get_probability(perm[0], perm[1:])
        line.append(record)
        line.append(prob)
        f.prob_table.append(line)
    return f


def generate_permutation(length):
    """
    Generate a permutation of True/False with given length
    """
    perms = set()
    for comb in itertools.combinations_with_replacement([True, False], length):
        for perm in itertools.permutations(comb):
            perms.add(perm)
    perms = list(perms)
    return perms

