# Linear Regression

# a linear model makes a prediction by simply computing a weighted
# sum of the input features, plus a constant called the bias term (also called the intercept
# term)
# y = θ0 + θ1*x1 + θ2*x2 + ⋯ + θn*xn
# y = θT*X

# Let’s generate some linear-looking data
import numpy as np
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()

# To find the value of θ that minimizes the cost function (MSE), there is a closed-form solution
# —in other words, a mathematical equation that gives the result directly. This is called
# the Normal Equation

# Now let’s compute θ using the Normal Equation
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best
# Out[4]:
# array([[4.13987882],
#        [2.86920484]])

# Now you can make predictions using θ
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
y_predict
# array([[4.13987882],
#        [9.87828851]])

plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.legend(loc="upper left", fontsize=14)
plt.show()

# The equivalent code using Scikit-Learn looks like this
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
# Out[11]: (array([4.13987882]), array([[2.86920484]]))

lin_reg.predict(X_new)
# array([[4.13987882],
#        [9.87828851]])



#%% Gradient Descent
# The general idea of Gradient Descent is to
# tweak parameters iteratively in order to minimize a cost function

# it measures the local gradient of the error function with regards to the
# parameter vector θ, and it goes in the direction of descending gradient. Once the gradient
# is zero, you have reached a minimum!

# When using Gradient Descent, you should ensure that all features
# have a similar scale

#%% Batch Gradient Descent
# you need to calculate
# how much the cost function will change if you change θj just a little bit. This is called
# a partial derivative

# formula involves calculations over the full training
# set X, at each Gradient Descent step! This is why the algorithm is
# called Batch Gradient Descent: it uses the whole batch of training
# data at every step.
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance

eta = 0.1 # learning rate
n_iterations = 1000
m = 100
theta = np.random.randn(2,1) # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

theta
# Out[5]:
# array([[3.78590608],
#        [3.13578118]])


#%% Stochastic Gradient Descent
# Stochastic Gradient Descent just
# picks a random instance in the training set at every step and computes the gradients
# based only on that single instance

n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1) # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

theta
# Out[12]:
# array([[3.8198601 ],
#        [3.09192007]])

# To perform Linear Regression using SGD with Scikit-Learn
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

sgd_reg.intercept_, sgd_reg.coef_
# Out[14]: (array([3.77506434]), array([3.12884666]))


#%% Mini-batch Gradient Descent
# at each step, instead of computing the gradients based on the full training
# set (as in Batch GD) or based on just one instance (as in Stochastic GD), Minibatch GD
# computes the gradients on small random sets of instances called minibatches

theta_path_mgd = []

n_iterations = 50
minibatch_size = 20

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

theta
# Out[16]:
# array([[3.91686646],
#        [3.28155626]])

#%% Polynomial Regression
# let’s generate some nonlinear data, based on a simple
# quadratic equation

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]
# Out[23]: array([2.38942838])
X_poly[0]
# Out[24]: array([2.38942838, 5.709368  ])

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
# Out[27]: (array([1.9735233]), array([[0.95038538, 0.52577032]]))
# the model estimates y = 0.5257*X^2 + 0.9503*X + 1.9735 when in fact the original
# function was y = 0.5*X^2 + 1.0*X + 2 + Gaussian noise

X_new=np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
plt.show()


#%% Learning curves
# to tell when a model is too simple or too complex.
# these are plots of the model’s performance
# on the training set and the validation set as a function of the training set size.
# To generate the plots, simply train the model several times on different sized subsets
# of the training set
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.axis([0, 80, 0, 3])

# Now let’s look at the learning curves of a 10th-degree polynomial model on the same
# data
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

plot_learning_curves(polynomial_regression, X, y)
plt.axis([0, 80, 0, 3])


#%% Regularized Linear Models
# For a linear model, regularization is typically achieved by constraining the weights of
# the model.

# Ridge Regression
# This forces the learning algorithm to not only fit the data but also keep the model
# weights as small as possible
# It is important to scale the data (e.g., using a StandardScaler)
# before performing Ridge Regression, as it is sensitive to the scale of
# the input features. This is true of most regularized models.
# Here is how to perform Ridge Regression with Scikit-Learn using a closed-form solution
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

# And using Stochastic Gradient Descent
sgd_reg = SGDRegressor(max_iter=5, penalty="l2", random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])


# Lasso Regression
# Least Absolute Shrinkage and Selection Operator Regression (simply called Lasso
# Regression) is another regularized version of Linear Regression: just like Ridge
# Regression, it adds a regularization term to the cost function, but it uses the ℓ1 norm
# of the weight vector instead of half the square of the ℓ2 norm
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])


# Elastic Net
# Elastic Net is a middle ground between Ridge Regression and Lasso Regression
# The regularization term is a simple mix of both Ridge and Lasso’s regularization terms,
# and you can control the mix ratio r. When r = 0, Elastic Net is equivalent to Ridge
# Regression, and when r = 1, it is equivalent to Lasso Regression
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])

# So when should you use Linear Regression, Ridge, Lasso, or Elastic Net? It is almost
# always preferable to have at least a little bit of regularization, so generally you should
# avoid plain Linear Regression. Ridge is a good default, but if you suspect that only a
# few features are actually useful, you should prefer Lasso or Elastic Net since they tend
# to reduce the useless features’ weights down to zero as we have discussed. In general,
# Elastic Net is preferred over Lasso since Lasso may behave erratically when the number
# of features is greater than the number of training instances or when several features
# are strongly correlated.


# Early Stopping
# A very different way to regularize iterative learning algorithms such as Gradient
# Descent is to stop training as soon as the validation error reaches a minimum. This is
# called early stopping.

from sklearn.preprocessing import StandardScaler

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler()),
    ])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1,
                       penalty=None,
                       eta0=0.0005,
                       warm_start=True,
                       learning_rate="constant",
                       random_state=42)

n_epochs = 500
train_errors, val_errors = [], []
for epoch in range(n_epochs):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    train_errors.append(mean_squared_error(y_train, y_train_predict))
    val_errors.append(mean_squared_error(y_val, y_val_predict))

best_epoch = np.argmin(val_errors)
best_val_rmse = np.sqrt(val_errors[best_epoch])

plt.annotate('Best model',
             xy=(best_epoch, best_val_rmse),
             xytext=(best_epoch, best_val_rmse + 1),
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=16,
            )

best_val_rmse -= 0.03  # just to make the graph look better
plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.show()

from sklearn.base import clone
sgd_reg = SGDRegressor(max_iter=1, warm_start=True, penalty=None,
                       learning_rate="constant", eta0=0.0005, random_state=42)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)



#%% Logistic Regression
# Logistic Regression (also called Logit Regression) is commonly
# used to estimate the probability that an instance belongs to a particular class

# Estimating Probabilities
# If the estimated probability is
# greater than 50%, then the model predicts that the instance belongs to that class
# (called the positive class, labeled “1”), or else it predicts that it does not (i.e., it
# belongs to the negative class, labeled “0”). This makes it a binary classifier.

# The logistic—also called the logit is a sigmoid function
# that outputs a number between 0 and 1
t = np.linspace(-10, 10, 100)
sig = 1 / (1 + np.exp(-t))
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
plt.axis([-10, 10, -0.1, 1.1])
plt.show()

# Training and Cost Function
# The cost function over the whole training set is simply the average cost over all training
# instances.
# this cost function is convex, so Gradient Descent (or any
# other optimization algorithm) is guaranteed to find the global minimum (if the learning
# rate is not too large and you wait long enough).

# Decision Boundaries
# Let’s try to build a classifier to detect the Iris-Virginica type based only on the petal
# width feature
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:, 3:] # petal width
y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0

# Now let’s train a Logistic Regression model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)

# Let’s look at the model’s estimated probabilities for flowers with petal widths varying
# from 0 to 3 cm
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()

decision_boundary
# Out[14]: array([1.61561562])
log_reg.predict([[1.7], [1.5]])
# Out[13]: array([1, 0])


# Softmax Regression
# to support multiple classes directly,
# without having to train and combine multiple binary classifiers
# The idea is quite simple: when given an instance x, the Softmax Regression model
# first computes a score sk(x) for each class k, then estimates the probability of each
# class by applying the softmax function (also called the normalized exponential) to the
# scores.

# Let’s use Softmax Regression to classify the iris flowers into all three classes
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)
softmax_reg.predict([[5, 2]])
# Out[16]: array([2])
softmax_reg.predict_proba([[5, 2]])
# Out[17]: array([[6.33134077e-07, 5.75276067e-02, 9.42471760e-01]])
