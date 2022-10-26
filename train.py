import csv

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_features_and_labels(file_name):
    with open(file_name, "r") as input_file:
        row_count = sum(1 for _ in input_file)

    features = np.ndarray((row_count, 1000))
    labels = np.ndarray(row_count, dtype=int)

    print("Loading embeddings..")
    with open(file_name, "r") as input_file:
        for i, row in enumerate(csv.reader(input_file)):
            features[i, :] = row[:1000]
            labels[i] = row[1001]

    return features, labels

training_features, training_labels = get_features_and_labels("embeddings.csv")

for split in [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]:
    X_train, X_test, y_train, y_test = train_test_split(training_features, training_labels, test_size=split, random_state=42, shuffle=True)
    print("Number of training examples: {}".format(len(X_train)))
    print("Number of test examples: {}".format(len(X_test)))
    clf = svm.SVC(random_state=42)
    clf = clf.fit(X_train, y_train)
    print("Training accuracy: {}".format(accuracy_score(y_train, clf.predict(X_train))))
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))
    print("Test accuracy: {}".format(test_accuracy))
    print("")
