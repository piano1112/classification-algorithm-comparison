import csv
import math
from collections import Counter


class Node:
    # cl is class for leaf node, or majority class for internal node
    # only internal node will have non-null attr_index and not empty children
    # children is a dict attribute value : child node
    def __init__(self, cl, attr_index=None, children=None):
        self.cl = cl
        self.attr_index = attr_index
        self.children = children or {}


def load_data(filename, has_class=True):
    data = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            if has_class:
                # list of attributes and class
                *attrs, cl = row
                data.append((attrs, cl))
            else:
                attrs = row
                data.append(attrs)
    return data


# return majority class of subset (examples)
# cls is list of classes
def mode(cls):
    counts = Counter(cls)
    if counts['yes'] >= counts['no']:
        return 'yes'
    else:
        return 'no'


# cls is list of classes
def entropy(cls):

    output = 0.0

    if not cls:
        return output

    counts = Counter(cls)
    for count in counts.values():
        prob = count / len(cls)
        output -= prob * math.log2(prob)
    return output


# examples is list of (list of attributes, class)
def choose_attribute(attr_indices, examples):

    # setup
    best_attr = None
    best_info_gain = -1
    h_examples = entropy([cl for _, cl in examples])

    # calculate information gain of each attribute
    for attr in attr_indices:

        # subsets is attribute value : list of classes
        subsets = {}
        for attrs, cl in examples:
            subsets.setdefault(attrs[attr], []).append(cl)
        
        info_gain = h_examples
        for subset in subsets.values():
            info_gain -= len(subset) / len(examples) * entropy(subset)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attr = attr

    return best_attr


# examples is a subset of training data 
# attr_indices is a list of attribute indices
# default is expected to be majority class of subset of parent node if specified
def build_dt(examples, attr_indices, default=None):

    # extract list of classes from examples (train_data)
    cls = [cl for _, cl in examples]

    # stopping criterion - stop (return a leaf node) if case a, b or c
    # case c) if examples is empty then return default
    if not examples:
        return Node(default)

    # case a) else if all examples have same classification then return the classification
    if all(cl == cls[0] for cl in cls):
        return Node(cls[0])

    # case b) else if attr_indices is empty then return mode(cls)
    if not attr_indices:
        return Node(mode(cls))

    # index of attribute with greatest information gain
    best_attr = choose_attribute(attr_indices, examples)

    # build tree by recursing 
    tree = Node(mode(cls), best_attr)
    values = set(attrs[best_attr] for attrs, _ in examples)
    for v_i in values:
        examples_i = [(attrs, cl) for attrs, cl in examples if attrs[best_attr] == v_i]
        subtree = build_dt(examples_i, [i for i in attr_indices if i != best_attr], mode(cls))
        tree.children[v_i] = subtree

    # return root node
    return tree


def dt_to_string(node, depth=0):
    indent = '  ' * depth

    # Leaf node
    if not node.children:
        return f"{indent}Leaf: {node.cl}\n"

    # Internal node
    lines = []
    lines.append(f"{indent}Internal: split on attribute[{node.attr_index}], majority_class={node.cl}\n")
    for value, child in node.children.items():
        lines.append(f"{indent}if attribute[{node.attr_index}] == {value}:\n")
        lines.append(dt_to_string(child, depth+1))
    return ''.join(lines)


def classify_example(tree, example):

    # If leaf node, return its class
    if not tree.children:
        return tree.cl

    # If internal node, recurse on one child node 
    value = example[tree.attr_index]
    # unseen (new) attribute value
    if value not in tree.children:
        return tree.cl
    return classify_example(tree.children[value], example)


def classify_dt(training_filename, testing_filename):
    
    # list of (list of attributes, class)
    train_data = load_data(training_filename, True)
    # list of list of attributes
    test_data = load_data(testing_filename, False)

    # build dt 
    # assign index 0, 1, ..., n-1 to n attributes
    attr_indices = list(range(len(train_data[0][0])))
    tree = build_dt(train_data, attr_indices)

    # print dt for report
    # print(dt_to_string(tree))

    # classify test data 
    results = []
    for example in test_data:
        cl = classify_example(tree, example)
        results.append(cl)

    return results


# Test
# classify_dt('train.csv', 'test.csv')
