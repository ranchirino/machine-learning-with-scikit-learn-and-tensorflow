# Voting Classifier
# Exercise: Load the MNIST data and split it into a training set, a validation set,
# and a test set (e.g., use 50,000 instances for training, 10,000 for validation, and 10,000 for testing).

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

from sklearn.model_selection import train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)

# Exercise: Then train various classifiers, such as a Random Forest classifier,
# an Extra-Trees classifier, and an SVM.
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

random_forest_clf = RandomForestClassifier(random_state=42)
extra_trees_clf = ExtraTreesClassifier(random_state=42)
svm_clf = LinearSVC(random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)

[estimator.score(X_val, y_val) for estimator in estimators]
# Out[3]: [0.9467, 0.9512, 0.8327, 0.9592]

# The linear SVM is far outperformed by the other classifiers.
# However, let's keep it for now since it may improve the voting classifier's performance.

# Exercise: Next, try to combine them into an ensemble that outperforms them all on the validation set,
# using a soft or hard voting classifier.

from sklearn.ensemble import VotingClassifier

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]

voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train, y_train)

voting_clf.score(X_val, y_val)
# Out[5]: 0.9645

[estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]
# Out[6]: [0.9467, 0.9512, 0.8327, 0.9592]

# Let's remove the SVM to see if performance improves.
# It is possible to remove an estimator by setting it to None using set_params() like this:
voting_clf.set_params(svm_clf=None)

# This updated the list of estimators:

voting_clf.estimators

# However, it did not update the list of trained estimators:
voting_clf.estimators_

# So we can either fit the VotingClassifier again,
# or just remove the SVM from the list of trained estimators:
del voting_clf.estimators_[2]

# Now let's evaluate the VotingClassifier again:
voting_clf.score(X_val, y_val)
# Out[11]: 0.9667

# The SVM was hurting performance.
# Now let's try using a soft voting classifier.
# We do not actually need to retrain the classifier, we can just set voting to "soft":
voting_clf.voting = "soft"
voting_clf.score(X_val, y_val)
# Out[13]: 0.971

# That's a significant improvement,
# and it's much better than each of the individual classifiers.

# Once you have found one, try it on the test set.
# How much better does it perform compared to the individual classifiers?
voting_clf.score(X_test, y_test)
# Out[14]: 0.9673

[estimator.score(X_test, y_test) for estimator in voting_clf.estimators_]
# Out[15]: [0.9434, 0.9444, 0.9556]

