from csv import reader
from math import exp, sqrt, pi
from random import randrange, seed


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


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


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
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


class NaiveBayes:

    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, n_folds):
        folds = cross_validation_split(self.dataset, n_folds)
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
            predicted = self.naive_bayes(train_set, test_set)
            actual = [row[-1] for row in fold]
            accuracy = calculate_accuracy(actual, predicted)
            scores.append(accuracy)
        return scores

    def naive_bayes(self, train, test):
        summarize = self.information_by_class(train)
        predictions = list()
        for row in test:
            output = self.predict(summarize, row)
            predictions.append(output)
        return predictions

    def predict(self, summaries, row):
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    def separate_by_class(self, dataset):
        """دیتاست ورودی را براساس کلاس ها به صورت دیکشنری ذخیره میکنیم"""
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if class_value not in separated:
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated

    # Calculate the mean, standard deviation and count for each column in a dataset
    def information_dataset(self, rows):
        summaries = [(mean(column), calculate_deviation(column), len(column)) for column in zip(*rows)]
        del (summaries[-1])
        return summaries

    # Split dataset by class then calculate statistics for each row
    def information_by_class(self, dataset):
        separated = self.separate_by_class(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.information_dataset(rows)
        return summaries

    # Calculate the Gaussian probability distribution function for x
    def calculate_probability(self, x, mean, standard_deviation):
        exponent = exp(-((x - mean) ** 2 / (2 * standard_deviation ** 2)))
        return (1 / (sqrt(2 * pi) * standard_deviation)) * exponent

    # Calculate the probabilities of predicting each class for a given row
    def calculate_class_probabilities(self, summaries, row):
        """به ازای هر کلاس ما احتمال داده ورودی به ان کلاس با استفاده از قانون بیزین حساب میکنیم
        P(class|data) = P(x|class) * P(class)
        """
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, standard_deviation, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, standard_deviation)
        return probabilities


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def calculate_deviation(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Test Naive Bayes on Iris Dataset
seed(1)
data = load_csv('iris.csv')
data = ConvertClassColumnToInt(ConvertStringColumnToFloat(data).do()).do()
scores = NaiveBayes(data).fit(n_folds=5)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
