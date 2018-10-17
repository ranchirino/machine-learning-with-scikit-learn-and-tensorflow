# Exercise: train an SVM regressor on the California housing dataset.

# Let's load the dataset using Scikit-Learn's fetch_california_housing() function:
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]

# Split it into a training set and a test set:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Don't forget to scale the data:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Let's train a simple LinearSVR first:
from sklearn.svm import LinearSVR

lin_svr = LinearSVR(random_state=42)
lin_svr.fit(X_train_scaled, y_train)

# Let's see how it performs on the training set:
from sklearn.metrics import mean_squared_error

y_pred = lin_svr.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_pred)
mse
# Out[5]: 0.9612806653297273

# Let's look at the RMSE:
np.sqrt(mse)
# Out[7]: 0.9804492160890983

# In this training set, the targets are tens of thousands of dollars.
# The RMSE gives a rough idea of the kind of error you should expect
# (with a higher weight for large errors):
# so with this model we can expect errors somewhere around $10,000.
# Not great. Let's see if we can do better with an RBF Kernel.
# We will use randomized search with cross validation to find the appropriate hyperparameter values for `C` and `gamma`:

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(SVR(), param_distributions, n_iter=10, verbose=2, random_state=42)
rnd_search_cv.fit(X_train_scaled, y_train)

rnd_search_cv.best_estimator_
# Out[9]:
# SVR(C=4.745401188473625, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
#   gamma=0.07969454818643928, kernel='rbf', max_iter=-1, shrinking=True,
#   tol=0.001, verbose=False)

# Now let's measure the RMSE on the training set:
y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_pred)
np.sqrt(mse)
# Out[10]: 0.5727524770785357

# Looks much better than the linear model.
# Let's select this model and evaluate it on the test set:

y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
np.sqrt(mse)
# Out[11]: 0.5929168385528746

