import pprint

import numpy as np
import pandas as pd
from sklearn import datasets


def calculate_entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = 0
    for i in range(len(elements)):
        entropy += (-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
    return entropy


def gain(data, source_feature, target_feature):
    n_entropy = 0
    values, counts = np.unique(data[source_feature], return_counts=True)
    for i in range(len(values)):
        entropy = calculate_entropy(data.where(data[source_feature] == values[i]).dropna()[target_feature])
        n_entropy += (counts[i] / np.sum(counts)) * entropy
    return n_entropy


def information_gain(df, s_feature, y):
    total_entropy = calculate_entropy(df[y])
    return total_entropy - gain(df, s_feature, y)


def split_information(data, f):
    elements, counts = np.unique(data[f], return_counts=True)
    entropy = 0
    for i in range(len(elements)):
        entropy += (-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
    return entropy


def gain_ratio(info_gain, data, best_feature):
    ratio = info_gain / split_information(data, best_feature)
    return ratio


def decision_tree(data, original_data, features, target_feature="target", parent=None, level=0):
    if (len(np.unique(
            data[target_feature])) <= 1):  # checking if all the values are same if yes then we reached at a leaf

        elements, counts = np.unique(data[target_feature], return_counts=True)
        print('Level ', level)
        if elements == 0:
            print('Count of 0 =', np.sum(counts))
        elif elements == 1:
            print('Count of 1 =', np.sum(counts))
        elif elements == 2:
            print('Count of 2 =', np.sum(counts))
        print('Current Entropy is =', calculate_entropy(data[target_feature]))
        print('Reached Leaf Node ')
        print()
        return np.unique(data[target_feature])[0]

    elif len(data) == 0:  # checking the data is empty or not

        return np.unique(original_data[target_feature])

    elif len(features) == 0:

        return parent

    else:

        parent_node = np.unique(data[target_feature])  # put all the unique values of target in parent node

        values = []
        for ftr in features:  # loop over all the features
            v = information_gain(data, ftr, target_feature)  # getting list of information gain of all features
            values.append(v)

        best_feature_index = np.argmax(
            values)  # taking out the index of the feature which contains max information gain
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}  # i have used dictionaries to show my actual tree

        tot_entropy = calculate_entropy(data[target_feature])  # calculated entropy at current node

        rat = gain_ratio(max(values), data,
                         best_feature)  # calculated gain ratio  of the features on which we split up on

        elements, counts = np.unique(data[target_feature], return_counts=True)
        print('Level ', level)  # these all are printing task
        for i in range(len(elements)):
            if elements[i] == 0:
                print('count of 0  =', counts[i])
            elif elements[i] == 1:
                print('count of 1  =', counts[i])
            elif elements[i] == 2:
                print('count of 2  =', counts[i])

        print('Current entropy is   = ', tot_entropy)
        print('Splitting on feature ', best_feature, ' with gain ratio ', rat)

        print()

        new_features = features  # ---> from here to
        features = []
        for i in new_features:
            #  (process to remove the feature from feature list after split
            if i != best_feature:
                features.append(i)
        level += 1
        new_features = None  # ---> to here

        for vals in np.unique(data[best_feature]):  # recursion of all different values in that splitting feature

            value = vals
            sub_data = (data[data[best_feature] == value]).dropna()

            subtree = decision_tree(sub_data, data, features, target_feature, parent_node, level)
            tree[best_feature][value] = subtree

        return tree


def label(val, *boundaries):
    if val < boundaries[0]:
        return 'a'
    elif val < boundaries[1]:
        return 'b'
    elif val < boundaries[2]:
        return 'c'
    else:
        return 'd'


# Function to convert a continuous data into labelled data
# There are 4 labels  - a, b, c, d
def to_label(df, old_feature_name):
    second = df[old_feature_name].mean()
    minimum = df[old_feature_name].min()
    first = (minimum + second) / 2
    maximum = df[old_feature_name].max()
    third = (maximum + second) / 2
    return df[old_feature_name].apply(label, args=(first, second, third))


def format_data(t, s):
    if not isinstance(t, dict) and not isinstance(t, list):
        print("\t" * s + str(t))
    else:
        for key in t:
            print("\t" * s + str(key))
            if not isinstance(t, list):
                format_data(t[key], s + 1)


def load_data():
    iris = datasets.load_iris()

    df = pd.DataFrame(iris.data)
    df.columns = ["sl", "sw", 'pl', 'pw']
    y = pd.DataFrame(iris.target)
    y.columns = ['target']
    # Convert all columns to labelled data
    df['sl_labeled'] = to_label(df, 'sl')
    df['sw_labeled'] = to_label(df, 'sw')
    df['pl_labeled'] = to_label(df, 'pl')
    df['pw_labeled'] = to_label(df, 'pw')

    df.head()
    df.drop(['sl', 'sw', 'pl', 'pw'], axis=1, inplace=True)
    df['target'] = y  # here i added the target column  in the data
    print(df.columns[:-1])  # this gives me the list of all features except target one
    return df


df = load_data()
tree = decision_tree(df, df, df.columns[:-1])  # print steps
print()
pprint.pprint(tree)
format_data(tree, 0)  # print tree
