# Exercise: Load the MNIST dataset and split it into a training set and a test set
# (take the first 60,000 instances for training, and the remaining 10,000 for testing).

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

X_train = mnist['data'][:60000]
y_train = mnist['target'][:60000]

X_test = mnist['data'][60000:]
y_test = mnist['target'][60000:]

# Exercise: Train a Random Forest classifier on the dataset and time how long it takes,
# then evaluate the resulting model on the test set.

from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(random_state=42)

import time
t0 = time.time()
rnd_clf.fit(X_train, y_train)
t1 = time.time()
print("Training took {:.2f}s".format(t1 - t0))
# Training took 9.08s

from sklearn.metrics import accuracy_score

y_pred = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred)
# Out[4]: 0.9455

# Exercise: Next, use PCA to reduce the dataset's dimensionality,
# with an explained variance ratio of 95%.

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)

# Exercise: Train a new Random Forest classifier on the reduced dataset and see
# how long it takes. Was training much faster?
rnd_clf2 = RandomForestClassifier(random_state=42)
t0 = time.time()
rnd_clf2.fit(X_train_reduced, y_train)
t1 = time.time()
print("Training took {:.2f}s".format(t1 - t0))
# Training took 21.93s

# Oh no! Training is actually more than twice slower now! How can that be? Well,
# as we saw in this chapter, dimensionality reduction does not always lead to faster
# training time: it depends on the dataset, the model and the training algorithm.
# If you try a softmax classifier instead of a random forest classifier,
# you will find that training time is reduced by a factor of 3 when using PCA.
# Actually, we will do this in a second, but first let's check the precision of the
# new random forest classifier.

# Exercise: Next evaluate the classifier on the test set:
# how does it compare to the previous classifier?
X_test_reduced = pca.transform(X_test)

y_pred = rnd_clf2.predict(X_test_reduced)
accuracy_score(y_test, y_pred)
# Out[7]: 0.8908

# It is common for performance to drop slightly when reducing dimensionality,
# because we do lose some useful signal in the process.
# However, the performance drop is rather severe in this case.
# So PCA really did not help: it slowed down training and reduced performance.

# Let's see if it helps when using softmax regression:

from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
t0 = time.time()
log_clf.fit(X_train, y_train)
t1 = time.time()
print("Training took {:.2f}s".format(t1 - t0))
# Training took 26.73s

y_pred = log_clf.predict(X_test)
accuracy_score(y_test, y_pred)
# Out[9]: 0.9252

# Okay, so softmax regression takes much longer to train on this dataset
# than the random forest classifier, plus it performs worse on the test set.
# But that's not what we are interested in right now,
# we want to see how much PCA can help softmax regression.
# Let's train the softmax regression model using the reduced dataset:

log_clf2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
t0 = time.time()
log_clf2.fit(X_train_reduced, y_train)
t1 = time.time()
print("Training took {:.2f}s".format(t1 - t0))
# Training took 10.87s

# Nice! Reducing dimensionality led to a 4× speedup. :) Let's the model's accuracy:

y_pred = log_clf2.predict(X_test_reduced)
accuracy_score(y_test, y_pred)
# Out[12]: 0.9198

# A very slight drop in performance, which might be a reasonable price to pay for a 4× speedup, depending on the application.
#
# So there you have it: PCA can give you a formidable speedup... but not always!
