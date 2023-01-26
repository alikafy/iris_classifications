from abc import ABC
from csv import reader
from math import sqrt
from random import randrange, seed


class PerProcessingData:

    def __init__(self, dataset):
        self.dataset = dataset

    def do(self):
        raise NotImplemented


class ConvertStringColumnToFloat(PerProcessingData):
    def do(self):
        for column in range(len(self.dataset[0]) - 1):
            for row in self.dataset:
                row[column] = float(row[column].strip())
        return self.dataset


class ConvertClassColumnToInt(PerProcessingData):
    def do(self):
        column = len(self.dataset[0]) - 1
        class_values = set([row[column] for row in self.dataset])
        lookup = dict()
        for i, value in enumerate(class_values):
            lookup[value] = i
        for row in self.dataset:
            row[column] = lookup[row[column]]
        return self.dataset


class MinMax(PerProcessingData):
    """Find the min and max values for each column"""

    def do(self):
        minmax = list()
        for i in range(len(self.dataset[0])):
            col_values = [row[i] for row in data]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax


def cross_validation(dataset, k_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / k_folds)
    for _ in range(k_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


class Distance(ABC):
    """ interface for calculate the distance between two vectors"""

    def algorithm(self, first_row, second_row):
        raise NotImplemented


class EuclideanDistance(Distance):
    def algorithm(self, first_row, second_row):
        distance = 0.0
        for i in range(len(first_row) - 1):
            distance += (first_row[i] - second_row[i]) ** 2
        return sqrt(distance)


class Knn:

    def __init__(self, dataset, number_of_neighbors: int, distance_algorithm: Distance):
        self.dataset = dataset
        self.number_of_neighbors = number_of_neighbors
        self.distance_algorithm = distance_algorithm

    def fit(self, n_folds):
        folds = cross_validation(self.dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = self.k_nearest_neighbors(train_set, test_set)
            actual = [row[-1] for row in fold]
            accuracy = calculate_accuracy(actual, predicted)
            scores.append(accuracy)
        return scores

    def k_nearest_neighbors(self, train, test):
        predictions = list()
        for row in test:
            output = self.predict(train, row)
            predictions.append(output)
        return predictions

    def get_neighbors(self, train, test_row):
        """Locate the most similar neighbors"""
        distances = list()
        for train_row in train:
            dist = self.distance_algorithm.algorithm(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.number_of_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    def predict(self, train, test_row):
        neighbors = self.get_neighbors(train, test_row)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


seed(1)
data = load_csv('iris.csv')
data = ConvertClassColumnToInt(ConvertStringColumnToFloat(data).do()).do()
scores = Knn(dataset=data, number_of_neighbors=5, distance_algorithm=EuclideanDistance()).fit(n_folds=5)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
