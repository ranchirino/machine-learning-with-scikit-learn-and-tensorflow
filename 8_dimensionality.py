# Apart from speeding up training, dimensionality reduction is also extremely useful
# for data visualization (or DataViz). Reducing the number of dimensions down to two
# (or three) makes it possible to plot a high - dimensional training set on a graph and
# often gain some important insights by visually detecting patterns, such as clusters.

#%% PCA
# Principal Component Analysis (PCA) is by far the most popular dimensionality reduction
# algorithm. First it identifies the hyperplane that lies closest to the data, and then
# it projects the data onto it.

# Preserving the Variance
# Before you can project the training set onto a lower-dimensional hyperplane, you
# first need to choose the right hyperplane.
# It seems reasonable to select the axis that preserves the maximum amount of variance,
# as it will most likely lose less information than the other projections.
# Another way to justify this choice is that it is the axis that minimizes the mean squared distance
# between the original dataset and its projection onto that axis. This is the rather
# simple idea behind PCA

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

# Singular Value Decomposition
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

# Once you have identified all the principal components, you can reduce the dimensionality
# of the dataset down to d dimensions by projecting it onto the hyperplane
# defined by the first d principal components

W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
X2D_using_svd = X2D


# With Scikit-Learn, PCA is really trivial. It even takes care of mean centering for you
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax = fig.add_subplot(122)
ax.scatter(X2D[:, 0], X2D[:, 1])
plt.show()


# Explained Variance Ratio
# Indica la proporción de la varianza del conjunto de datos
# que se encuentra a lo largo del eje de cada componente principal.
print(pca.explained_variance_ratio_)
# [0.84248607 0.14631839]
# This tells you that 84.2% of the dataset’s variance lies along the first axis, and 14.6%
# lies along the second axis. This leaves less than 1.2% for the third axis, so it is reasonable
# to assume that it probably carries little information.


# Choosing the Right Number of Dimensions
# Instead of arbitrarily choosing the number of dimensions to reduce down to, it is
# generally preferable to choose the number of dimensions that add up to a sufficiently
# large portion of the variance (e.g., 95%).

# The following code computes PCA without reducing dimensionality, then computes
# the minimum number of dimensions required to preserve 95% of the training set’s
# variance
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
# Out[3]: 2

# You could then set n_components=d and run PCA again. However, there is a much
# better option: instead of specifying the number of principal components you want to
# preserve, you can set n_components to be a float between 0.0 and 1.0, indicating the
# ratio of variance you wish to preserve

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
pca.n_components_
# Out[5]: 2
np.sum(pca.explained_variance_ratio_)
# Out[6]: 0.988804464429311

# Yet another option is to plot the explained variance as a function of the number of
# dimensions (simply plot cumsum)


# PCA for Compression
# the following code compresses the MNIST dataset down to 154 dimensions, then uses
# the inverse_transform() method to decompress it back to 784 dimensions.
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

from sklearn.model_selection import train_test_split

X = mnist["data"]
y = mnist["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)

pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
# Out[11]: 154

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
pca.n_components_
# Out[14]: 154

np.sum(pca.explained_variance_ratio_)
# Out[15]: 0.9504463030200186

pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.title("Original", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::2100])
plt.title("Compressed", fontsize=16)

X_reduced_pca = X_reduced


# Incremental PCA
# you can split the training
# set into mini-batches and feed an IPCA algorithm one mini-batch at a time. This is
# useful for large training sets, and also to apply PCA online

# The following code splits the MNIST dataset into 100 mini-batches (using NumPy’s
# array_split() function) and feeds them to Scikit-Learn’s IncrementalPCA class to
# reduce the dimensionality of the MNIST dataset down to 154 dimensions
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    print(".", end="") # not shown in the book
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced)

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.subplot(122)
plot_digits(X_recovered_inc_pca[::2100])
plt.tight_layout()

# Using memmap()
filename = "my_mnist.data"
m, n = X_train.shape

X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))
X_mm[:] = X_train

# Now deleting the memmap() object will trigger its Python finalizer,
# which ensures that the data is saved to disk.
del X_mm

X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))

batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)

# Randomized PCA
# it is dramatically faster than the previous algorithms when d is much
# smaller than n.
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)


#%% Kernel PCA
from sklearn.datasets import make_swiss_roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

# shows the Swiss roll, reduced to two dimensions using a linear kernel
# (equivalent to simply using the PCA class), an RBF kernel, and a sigmoid kernel
# (Logistic).
lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

y = t > 6.9

plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"),
                            (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
    X_reduced = pca.fit_transform(X)
    if subplot == 132:
        X_reduced_rbf = X_reduced

    plt.subplot(subplot)
    # plt.plot(X_reduced[y, 0], X_reduced[y, 1], "gs")
    # plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], "y^")
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

plt.show()


# Selecting a Kernel and Tuning Hyperparameters
# the following
# code creates a two-step pipeline, first reducing dimensionality to two dimensions
# using kPCA, then applying Logistic Regression for classification. Then it uses Grid
# SearchCV to find the best kernel and gamma value for kPCA in order to get the best
# classification accuracy at the end of the pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)

# The best kernel and hyperparameters are then available through the best_params_
# variable
print(grid_search.best_params_)
# {'kpca__gamma': 0.043333333333333335, 'kpca__kernel': 'rbf'}


rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
                    fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

from sklearn.metrics import mean_squared_error

mean_squared_error(X, X_preimage)
# Out[9]: 32.78630879576615


#%% LLE
# Locally Linear Embedding (LLE) is another very powerful nonlinear dimensionality
# reduction (NLDR) technique. It is a Manifold Learning technique that does not rely
# on projections like the previous algorithms.
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)

from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(X)

plt.title("Unrolled swiss roll using LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)
plt.show()


#%% Other Dimensionality Reduction Techniques
# MDS, Isomap and t-SNE

from sklearn.manifold import MDS
mds = MDS(n_components=2, random_state=42)
X_reduced_mds = mds.fit_transform(X)

from sklearn.manifold import Isomap
isomap = Isomap(n_components=2)
X_reduced_isomap = isomap.fit_transform(X)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne.fit_transform(X)


titles = ["MDS", "Isomap", "t-SNE"]

plt.figure(figsize=(11,4))

for subplot, title, X_reduced in zip((131, 132, 133), titles,
                                     (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)):
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

plt.show()

