# Exercise: train and fine-tune a Decision Tree for the moons dataset.

# a. Generate a moons dataset using make_moons(n_samples=10000, noise=0.4).
# Adding random_state=42 to make this notebook's output constant:

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

# b. Split it into a training set and a test set using train_test_split().
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# c. Use grid search with cross-validation (with the help of the GridSearchCV class)
# to find good hyperparameter values for a DecisionTreeClassifier.
# Hint: try various values for max_leaf_nodes.

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=1, verbose=1)

grid_search_cv.fit(X_train, y_train)

grid_search_cv.best_estimator_
# Out[2]:
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#             max_features=None, max_leaf_nodes=17,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=42,
#             splitter='best')

# d. Train it on the full training set using these hyperparameters,
# and measure your model's performance on the test set.
# You should get roughly 85% to 87% accuracy.

# By default, GridSearchCV trains the best model found on the whole training set
# (you can change this by setting refit=False),
# so we don't need to do it again. We can simply evaluate the model's accuracy:

from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)
# Out[3]: 0.8695

