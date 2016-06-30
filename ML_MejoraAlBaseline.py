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
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
             "Quadratic Discriminant Analysis"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]



    datasets = np.array([X_tr,y_tr.ix[:,0]])

    i = 1
    # iterate over datasets, despues sobre los topicos
    for ds in datasets:
        # preprocess dataset, split into training and test part
        X_train, y_train = ds
        #X = StandardScaler().fit_transform(X)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

        X_test = X_te
        y_test = y_te.ix[:,0]

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)

            score = clf.score(X_test, y_test)

            pr, rc, fb, su = precision_recall_fscore_support(y_test, score, average='macro')

            print("Clasificador {0}, Precision {1} Recall {2} F1 {3}".format(name, pr, rc, fb))
