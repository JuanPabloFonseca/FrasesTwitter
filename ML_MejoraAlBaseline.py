import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import precision_recall_fscore_support

h = .02  # step size in the mesh


def clasificadores_supervisados(X_tr, y_tr, X_te, y_te):
    names = [
         "Nearest Neighbors",
         "Linear SVM",
         "RBF SVM",
         "Decision Tree",
        # "Naive Bayes"
    ]


    resultado = []

    # iterate over topics
    for i in range(0, len(y_tr.columns)):

        X_train = X_tr.toarray()
        y_train = y_tr.ix[:, i]

        X_test = X_te.toarray()
        y_test = y_te.ix[:, i]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="poly", degree=3),
            SVC(gamma=2, C=1),
            DecisionTreeClassifier(max_depth=5),
            # GaussianNB()
        ]

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)

            # score = clf.score(X_test)
            score = clf.predict(X_test)

            pr, rc, fb, su = precision_recall_fscore_support(y_test, score, average='binary') # macro

            print("Topico {0}-Clasificador {1}: Precision {2} Recall {3} F1 {4}".format(i, name, pr, rc, fb))

        resultado.append(classifiers)