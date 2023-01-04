import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC


def plot_iris_data():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    X_reduced = PCA(n_components=3).fit_transform(iris.data)
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=y,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )

    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.zaxis.set_ticklabels([])

    plt.show()


def load_data_pandas():
    column_names = ["sepal_length_in_cm", "sepal_width_in_cm", "petal_length_in_cm", "petal_width_in_cm", "class"]
    dataset = pd.read_csv("iris.data", header=None, names=column_names)

    print(dataset.head())
    return dataset


def encoding_target_data(dataset):
    return dataset.replace({"class": {"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}})


def split_data(dataset):
    x = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    return x_train, x_test, y_train, y_test


def svm_train(x_train, y_train):
    classifier = SVC(kernel="linear", random_state=0)
    classifier.fit(x_train, y_train)
    return classifier


def predict(classifier, x_test):
    y_pred = classifier.predict(x_test)
    return y_pred


def validation(x_train, y_train, y_test, y_pred, classifier):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
    print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))


plot_iris_data()
dataset = load_data_pandas()
dataset = encoding_target_data(dataset)
x_train, x_test, y_train, y_test = split_data(dataset)
classifier = svm_train(x_train, y_train)
y_pred = predict(classifier, x_test)
print("predictions: ", y_pred)
validation(x_train, y_train, y_test, y_pred, classifier)
