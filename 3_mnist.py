# El siguiente código recupera el conjunto de datos MNIST
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

X, y = mnist["data"], mnist["target"]
# X.shape
# y.shape

#%% Echemos un vistazo a un dígito del conjunto de datos
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")

# y[36000]

#%% You should always create a test set and set it aside before inspecting the data closely
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Let’s also shuffle the training set
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#%% Training a Binary Classifier
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

# now let’s pick a classifier and train it. A good place to start is with a Stochastic
# Gradient Descent (SGD) classifier
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Now you can use it to detect images of the number 5
sgd_clf.predict([some_digit])

#%% Performance Measures
# Measuring Accuracy Using Cross-Validation
# Remember that K-fold crossvalidation
# means splitting the training set into K-folds (in this case, three), then making
# predictions and evaluating them on each fold using a model trained on the
# remaining folds
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# Out[28]: array([0.963 , 0.9673, 0.9649])
# Above 95% accuracy

# veamos un clasificador muy tonto que solo clasifica cada imagen en la clase "no-5"
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# Out[32]: array([0.9077 , 0.91095, 0.9103 ])
# That’s right, it has over 90% accuracy! This is simply because only about 10% of the
# images are 5s, so if you always guess that an image is not a 5, you will be right about
# 90% of the time.

#%% Confusion Matrix
# La idea general es contar el número de veces que las instancias de la clase A se clasifican como clase B.
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
# Out[36]:
# array([[53843,   736],
#        [ 1360,  4061]], dtype=int64)

# Each row in a confusion matrix represents an actual class, while each column represents
# a predicted class
# The first row of this matrix considers non-5 images (the negative
# class): 53543 of them were correctly classified as non-5s (they are called true
# negatives), while the remaining 736 were wrongly classified as 5s (false positives).
# The second row considers the images of 5s (the positive class): 1360 were wrongly
# classified as non-5s (false negatives), while the remaining 4061 were correctly classified
# as 5s (true positives)

# Precision and Recall
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
# Out[38]: 0.8465707734000417
recall_score(y_train_5, y_train_pred)
# Out[39]: 0.7491237779007563

# Cuando afirma que una imagen representa un 5, es correcta solo el 84% del tiempo.
# Además, solo detecta el 74% de los 5s.

# It is often convenient to combine precision and recall into a single metric called the F1
# score, in particular if you need a simple way to compare two classifiers. The F1 score is
# the harmonic mean of precision and recall
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)
# Out[40]: 0.7948717948717948


# Precision/Recall Tradeoff

# Instead of calling the classifier’s
# predict() method, you can call its decision_function() method, which returns a
# score for each instance, and then make predictions based on those scores using any
# threshold you want
y_scores = sgd_clf.decision_function([some_digit])
y_scores
# Out[4]: array([110733.94625458])
threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
# Out[6]: array([ True])

threshold = 200000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
# Out[7]: array([False])

# This confirms that raising the threshold decreases recall. The image actually represents
# a 5, and the classifier detects it when the threshold is 0, but it misses it when the
# threshold is increased to 200,000.

# So how can you decide which threshold to use?
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

# Now with these scores you can compute precision and recall for all possible thresholds
# using the precision_recall_curve() function

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.title("Precision and recall versus the decision threshold")
    plt.grid()

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# Now you can simply select the threshold value that gives you the best precision/recall
# tradeoff for your task

# So let’s suppose you decide to aim for 90% precision. You look up the first plot
# (zooming in a bit) and find that you need to use a threshold of about 51309. To make
# predictions (on the training set for now), instead of calling the classifier’s predict()
# method, you can just run this code
y_train_pred_90 = (y_scores > 51309)
precision_score(y_train_5, y_train_pred_90)
# Out[23]: 0.8999016232169208
recall_score(y_train_5, y_train_pred_90)
# Out[24]: 0.6749677181331858

#%% The receiver operating characteristic (ROC)
# the ROC curve plots the true positive rate (another name
# for recall) against the false positive rate
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("FPR against the TPR")
    plt.grid()

plot_roc_curve(fpr, tpr)
plt.show()

# One way to compare classifiers is to measure the area under the curve (AUC).
# A perfect classifier will have a ROC AUC equal to 1
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
# Out[28]: 0.9643213182731702

# Let’s train a RandomForestClassifier and compare its ROC curve and ROC AUC
# score to the SGDClassifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()
# the RandomForestClassifier’s ROC curve looks much
# better than the SGDClassifier’s: it comes much closer to the top-left corner

roc_auc_score(y_train_5, y_scores_forest)
# Out[35]: 0.9930241854404718


#%% Multiclass Classification
sgd_clf.fit(X_train, y_train) # y_train, not y_train_5
sgd_clf.predict([some_digit])
# Out[36]: array([5.])

some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores
# Out[37]:
# array([[-144544.32935261, -468242.9475172 , -402736.0614126 ,
#         -224430.32617788, -464528.52072681,  110733.94625458,
#         -567203.44947276, -472713.16588395, -683068.6746175 ,
#         -494461.82529041]])
# The highest score is indeed the one corresponding to class 5

np.argmax(some_digit_scores)
# Out[38]: 5
sgd_clf.classes_
# Out[39]: array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
sgd_clf.classes_[5]
# Out[41]: 5.0

from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
# Out[42]: array([5.])
len(ovo_clf.estimators_)
# Out[43]: 45

# Training a RandomForestClassifier is just as easy
# from sklearn.ensemble import RandomForestClassifier
# forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
# Out[44]: array([5.])
forest_clf.predict_proba([some_digit])
# Out[45]: array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# Out[46]: array([0.8704759 , 0.85054253, 0.84927739])

# simply scaling the inputs increases accuracy above 91%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
# Out[47]: array([0.91106779, 0.90869543, 0.91153673])


#%% Error Analysis
# First, you can look at the confusion matrix. You need to make predictions using the
# cross_val_predict() function, then call the confusion_matrix() function
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# Let’s focus the plot on the errors. First, you need to divide each value in the confusion
# matrix by the number of images in the corresponding class, so you can compare error
# rates instead of absolute number of errors
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# Now let’s fill the diagonal with zeros to keep only the errors, and let’s plot the result
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

# Looking at this plot, it seems that your efforts should be spent on improving
# classification of 8s and 9s, as well as fixing the specific 3/5 confusion
# let’s plot examples of 3s and 5s
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

def plot_digits(instances, images_per_row=10, **options):
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


plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()


#%% Multilabel Classification
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

# This code creates a y_multilabel array containing two target labels for each digit
# image: the first indicates whether or not the digit is large (7, 8, or 9) and the second
# indicates whether or not it is odd
knn_clf.predict([some_digit])
# Out[20]: array([[False,  True]])

# This code computes the average F1 score across all labels
# Warning: the following cell may take a very long time (possibly hours depending on your hardware).
# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
# f1_score(y_train, y_train_knn_pred, average="macro")

#%% Multioutput Classification
# Let’s start by creating the training and test sets by taking the MNIST images and
# adding noise to their pixel intensities using NumPy’s randint() function. The target
# images will be the original images
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test

some_index = 5500
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
plt.show()

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
