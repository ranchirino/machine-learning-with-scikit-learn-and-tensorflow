# Exercise: train a LinearSVC on a linearly separable dataset.
# Then train an SVC and a SGDClassifier on the same dataset.
# See if you can get them to produce roughly the same model.

# Usemos el conjunto de datos Iris:
# las clases Iris Setosa e Iris Versicolor son linealmente separables.

import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()

iris["feature_names"]
# Out[7]:
# ['sepal length (cm)',
#  'sepal width (cm)',
#  'petal length (cm)',
#  'petal width (cm)']
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

iris["target_names"]
# Out[11]: array(['setosa', 'versicolor', 'virginica'], dtype='<U10')

# selecciono solo las setosa (y=0) o las versicolor (y=1)
setosa_or_versicolor = (y == 0) | (y == 1)

X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

C = 5
alpha = 1 / (C * len(X))

lin_clf = LinearSVC(loss="hinge", C=C, random_state=42)
svm_clf = SVC(kernel="linear", C=C)
sgd_clf = SGDClassifier(loss="hinge", learning_rate="constant", eta0=0.001, alpha=alpha,
                        max_iter=100000, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lin_clf.fit(X_scaled, y)
svm_clf.fit(X_scaled, y)
sgd_clf.fit(X_scaled, y)

print("LinearSVC:                   ", lin_clf.intercept_, lin_clf.coef_)
print("SVC:                         ", svm_clf.intercept_, svm_clf.coef_)
print("SGDClassifier(alpha={:.5f}):".format(sgd_clf.alpha), sgd_clf.intercept_, sgd_clf.coef_)
# LinearSVC:                    [0.28481447] [[1.05541976 1.09851597]]
# SVC:                          [0.31933577] [[1.1223101  1.02531081]]
# SGDClassifier(alpha=0.00200): [0.32] [[1.12293103 1.02620763]]

# Let's plot the decision boundaries of these three models:
# y = w0*x0 + w1*x1 + b
# w0 y w1 son los coeficientes y b el intercepto
# En el limite de decision de las dos clases, y = 0, por lo que
# 0 = w0*x0 + w1*x1 + b
# despejando x1 tenemos
# x1 = -(w0/w1)*x0 - b/w1, por lo que -w0/w1 es la pendiente de la linea de decision
# y -b/w1 es el bias

# Compute the slope and bias of each decision boundary
w1 = -lin_clf.coef_[0, 0]/lin_clf.coef_[0, 1]
b1 = -lin_clf.intercept_[0]/lin_clf.coef_[0, 1]

w2 = -svm_clf.coef_[0, 0]/svm_clf.coef_[0, 1]
b2 = -svm_clf.intercept_[0]/svm_clf.coef_[0, 1]

w3 = -sgd_clf.coef_[0, 0]/sgd_clf.coef_[0, 1]
b3 = -sgd_clf.intercept_[0]/sgd_clf.coef_[0, 1]

# Transform the decision boundary lines back to the original scale
line1 = scaler.inverse_transform([[-10, -10 * w1 + b1], [10, 10 * w1 + b1]])
line2 = scaler.inverse_transform([[-10, -10 * w2 + b2], [10, 10 * w2 + b2]])
line3 = scaler.inverse_transform([[-10, -10 * w3 + b3], [10, 10 * w3 + b3]])

# Plot all three decision boundaries
plt.figure(figsize=(11, 4))
plt.plot(line1[:, 0], line1[:, 1], "k:", label="LinearSVC")
plt.plot(line2[:, 0], line2[:, 1], "b--", linewidth=2, label="SVC")
plt.plot(line3[:, 0], line3[:, 1], "r-", label="SGDClassifier")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs") # label="Iris-Versicolor"
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo") # label="Iris-Setosa"
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper center", fontsize=14)
plt.axis([0, 5.5, 0, 2])
plt.show()