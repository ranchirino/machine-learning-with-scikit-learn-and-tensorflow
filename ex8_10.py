# Exercise: Use t-SNE to reduce the MNIST dataset down to two dimensions and
# plot the result using Matplotlib.
# You can use a scatterplot using 10 different colors to represent each image's target class.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Let's start by loading the MNIST dataset
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

# Dimensionality reduction on the full 60,000 images takes a very long time,
# so let's only do this on a random subset of 10,000 images:

np.random.seed(42)

m = 10000
idx = np.random.permutation(60000)[:m]

X = mnist['data'][idx]
y = mnist['target'][idx]

# Now let's use t-SNE to reduce dimensionality down to 2D so we can plot the dataset:
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

# Now let's use Matplotlib's scatter() function to plot a scatterplot,
# using a different color for each digit:
plt.figure(figsize=(13,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()

# Isn't this just beautiful?
# This plot tells us which numbers are easily distinguishable from the others
# (e.g., 0s, 6s, and most 8s are rather well separated clusters),
# and it also tells us which numbers are often hard to distinguish
# (e.g., 4s and 9s, 5s and 3s, and so on).

# Let's focus on digits 2, 3 and 5, which seem to overlap a lot.
plt.figure(figsize=(9,9))
cmap = matplotlib.cm.get_cmap("jet")
for digit in (2, 3, 5):
    plt.scatter(X_reduced[y == digit, 0], X_reduced[y == digit, 1], c=cmap(digit / 9))
plt.axis('off')
plt.show()

# Let's see if we can produce a nicer image by running t-SNE on these 3 digits:
idx = (y == 2) | (y == 3) | (y == 5)
X_subset = X[idx]
y_subset = y[idx]

tsne_subset = TSNE(n_components=2, random_state=42)
X_subset_reduced = tsne_subset.fit_transform(X_subset)

plt.figure(figsize=(9,9))
for digit in (2, 3, 5):
    plt.scatter(X_subset_reduced[y_subset == digit, 0], X_subset_reduced[y_subset == digit, 1], c=cmap(digit / 9))
plt.axis('off')
plt.show()

# Much better, now the clusters have far less overlap.
# But some 3s are all over the place.
# Plus, there are two distinct clusters of 2s, and also two distinct clusters of 5s.
# It would be nice if we could visualize a few digits from each cluster,
# to understand why this is the case. Let's do that now.

# Exercise: Alternatively, you can write colored digits at the location of each instance,
# or even plot scaled-down versions of the digit images themselves
# (if you plot all digits, the visualization will be too cluttered,
# so you should either draw a random sample or plot an instance only
# if no other instance has already been plotted at a close distance).
# You should get a nice visualization with well-separated clusters of digits.

# Let's create a plot_digits() function that will draw a scatterplot
# (similar to the above scatterplots) plus write colored digits,
# with a minimum distance guaranteed between these digits.
# If the digit images are provided, they are plotted instead.
# This implementation was inspired from one of Scikit-Learn's excellent examples
# (plot_lle_digits, based on a different digit dataset).

from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = matplotlib.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=cmap(digit / 9), s=2)
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)

# Let's try it! First let's just write colored digits:
plot_digits(X_reduced, y)

# Well that's okay, but not that beautiful. Let's try with the digit images:
plot_digits(X_reduced, y, images=X, figsize=(35, 25))

plot_digits(X_subset_reduced, y_subset, images=X_subset, figsize=(22, 22))


# Exercise: Try using other dimensionality reduction algorithms
# such as PCA, LLE, or MDS and compare the resulting visualizations.
# Let's start with PCA. We will also time how long it takes:
from sklearn.decomposition import PCA
import time

t0 = time.time()
X_pca_reduced = PCA(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print("PCA took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_reduced, y)
plt.show()
# PCA took 0.8s.

# Wow, PCA is blazingly fast! But although we do see a few clusters,
# there's way too much overlap. Let's try LLE:
from sklearn.manifold import LocallyLinearEmbedding

t0 = time.time()
X_lle_reduced = LocallyLinearEmbedding(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print("LLE took {:.1f}s.".format(t1 - t0))
plot_digits(X_lle_reduced, y)
plt.show()
# LLE took 289.3s.

# That took a while, and the result does not look too good.
# Let's see what happens if we apply PCA first, preserving 95% of the variance:
from sklearn.pipeline import Pipeline

pca_lle = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("lle", LocallyLinearEmbedding(n_components=2, random_state=42)),
])
t0 = time.time()
X_pca_lle_reduced = pca_lle.fit_transform(X)
t1 = time.time()
print("PCA+LLE took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_lle_reduced, y)
plt.show()
# PCA+LLE took 107.4s.

# The result is more or less the same, but this time it was almost 4× faster.

# Let's try MDS. It's much too long if we run it on 10,000 instances,
# so let's just try 2,000 for now:
from sklearn.manifold import MDS

m = 2000
t0 = time.time()
X_mds_reduced = MDS(n_components=2, random_state=42).fit_transform(X[:m])
t1 = time.time()
print("MDS took {:.1f}s (on just 2,000 MNIST images instead of 10,000).".format(t1 - t0))
plot_digits(X_mds_reduced, y[:m])
plt.show()
# MDS took 265.7s (on just 2,000 MNIST images instead of 10,000).

# Meh. This does not look great, all clusters overlap too much.
# Let's try with PCA first, perhaps it will be faster?
from sklearn.pipeline import Pipeline

pca_mds = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("mds", MDS(n_components=2, random_state=42)),
])
t0 = time.time()
X_pca_mds_reduced = pca_mds.fit_transform(X[:2000])
t1 = time.time()
print("PCA+MDS took {:.1f}s (on 2,000 MNIST images).".format(t1 - t0))
plot_digits(X_pca_mds_reduced, y[:2000])
plt.show()
# PCA+MDS took 311.0s (on 2,000 MNIST images).

# Same result, and no speedup: PCA did not help (or hurt).

# Let's try LDA:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

t0 = time.time()
X_lda_reduced = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
t1 = time.time()
print("LDA took {:.1f}s.".format(t1 - t0))
plot_digits(X_lda_reduced, y, figsize=(12,12))
plt.show()
# LDA took 2.7s.

# This one is very fast, and it looks nice at first,
# until you realize that several clusters overlap severely.

# Well, it's pretty clear that t-SNE won this little competition, wouldn't you agree?
# We did not time it, so let's do that now:
from sklearn.manifold import TSNE

t0 = time.time()
X_tsne_reduced = TSNE(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print("t-SNE took {:.1f}s.".format(t1 - t0))
plot_digits(X_tsne_reduced, y)
plt.show()
# t-SNE took 975.6s.

# It's twice slower than LLE, but still much faster than MDS,
# and the result looks great. Let's see if a bit of PCA can speed it up:
pca_tsne = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("tsne", TSNE(n_components=2, random_state=42)),
])
t0 = time.time()
X_pca_tsne_reduced = pca_tsne.fit_transform(X)
t1 = time.time()
print("PCA+t-SNE took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_tsne_reduced, y)
plt.show()
# PCA+t-SNE took 690.2s.

# Yes, PCA roughly gave us a 25% speedup, without damaging the result. We have a winner!

