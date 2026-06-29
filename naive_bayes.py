import csv
import math


def load_data(filename, has_class=True):
    data = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            if has_class:
                # list of attributes and class
                *attrs, cl = row
                attrs = [float(x) for x in attrs]
                data.append((attrs, cl))
            else:
                attrs = [float(x) for x in row]
                data.append(attrs)
    return data


def separate_by_class(dataset):
    separated = {}
    for attrs, cl in dataset:
        separated.setdefault(cl, []).append(attrs)
    return separated


def get_mean_stdev(lists_attrs):
    # list of (mean, standard deviation) of each attributes
    mean_stdev = []
    # attr_values is list of values in a column (attribute)
    for values_col in zip(*lists_attrs):
        mean = sum(values_col) / len(values_col)
        stdev = math.sqrt(sum((x - mean) ** 2 for x in values_col) / len(values_col))
        mean_stdev.append((mean, stdev))
    return mean_stdev


def calculate_density(attr, mean, stdev):
    exponent = -((attr - mean) ** 2) / (2 * stdev ** 2)
    return ((math.e ** exponent) / (math.sqrt(2 * math.pi) * stdev)) 


def calculate_numerators(mean_stdevs, p_cls, attrs):
    # class : numerator
    numerators = {}
    for cl, mean_stdev in mean_stdevs.items():
        numerator = p_cls[cl]
        for i, (mean, stdev) in enumerate(mean_stdev):
            numerator *= calculate_density(attrs[i], mean, stdev)
        numerators[cl] = numerator
    return numerators


# perform Naive Bayes classification
def classify_nb(training_filename, testing_filename):
    # should be implemented for numeric attributes
    # use a probability density function for a normal distribution
    # if there is ever a tie between the two classes, choose class yes
    
    # list of (list of attributes, class)
    train_data = load_data(training_filename, True)
    # list of list of attributes
    test_data = load_data(testing_filename, False)

    # get mean and standard deviation of training dataset
    train_count = len(train_data)
    # (class, list of list of attributes in that class)
    separated = separate_by_class(train_data)
    # dict of class : P(class)
    p_cls = {}
    # dict of class : list of (mean, standard deviation) of each attribute
    mean_stdevs = {}
    for cl, lists_attrs in separated.items():
        mean_stdevs[cl] = get_mean_stdev(lists_attrs)
        p_cls[cl] = len(lists_attrs) / train_count

    # classify test data
    output = []
    for attrs in test_data:
        # dict of class : numerator
        numerators = calculate_numerators(mean_stdevs, p_cls, attrs) 
        if numerators["yes"] >= numerators["no"]:
            output.append("yes") 
        else:
            output.append("no")  

    return output
