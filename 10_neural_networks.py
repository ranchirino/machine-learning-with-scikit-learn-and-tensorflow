import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

#%% Perceptrons
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
y = (iris.target == 0).astype(np.int)  # Iris Setosa?

per_clf = Perceptron(max_iter=100, random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])
y_pred
# Out[5]: array([1])

a = -per_clf.coef_[0][0] / per_clf.coef_[0][1]
b = -per_clf.intercept_ / per_clf.coef_[0][1]

axes = [0, 5, 0, 2]

x0, x1 = np.meshgrid(
        np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
        np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]
y_predict = per_clf.predict(X_new)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs", label="Not Iris-Setosa")
plt.plot(X[y==1, 0], X[y==1, 1], "yo", label="Iris-Setosa")

plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-", linewidth=3)
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#9898ff', '#fafab0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.axis(axes)
plt.show()


#%% Activation functions
def logit(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)

z = np.linspace(-5, 5, 200)

plt.figure(figsize=(11,4))

plt.subplot(121)
plt.plot(z, np.sign(z), "r-", linewidth=2, label="Step")
plt.plot(z, logit(z), "g--", linewidth=2, label="Logit")
plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
plt.legend(loc="center right", fontsize=14)
plt.title("Activation functions", fontsize=14)
plt.axis([-5, 5, -1.2, 1.2])

plt.subplot(122)
plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Step")
plt.plot(0, 0, "ro", markersize=5)
plt.plot(0, 0, "rx", markersize=10)
plt.plot(z, derivative(logit, z), "g--", linewidth=2, label="Logit")
plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
#plt.legend(loc="center right", fontsize=14)
plt.title("Derivatives", fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])

plt.show()


#%% FNN for MNIST
# Training an MLP with TensorFlow’s High-Level API
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# the following
# code trains a DNN for classification with two hidden layers (one with 300
# neurons, and the other with 100 neurons) and a softmax output layer with 10
# neurons
feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], n_classes=10,
                                     feature_columns=feature_cols)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)
dnn_clf.train(input_fn=input_fn)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results = dnn_clf.evaluate(input_fn=test_input_fn)
eval_results
# Out[15]:
# {'accuracy': 0.9781,
#  'average_loss': 0.10410681,
#  'global_step': 44000,
#  'loss': 13.178078}

y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
y_pred = list(y_pred_iter)
y_pred[0]
# Out[17]:
# {'class_ids': array([7], dtype=int64),
#  'classes': array([b'7'], dtype=object),
#  'logits': array([ -8.455025 ,  -3.297753 ,  -4.6632957,   3.888749 , -10.384299 ,
#          -8.830646 , -20.407845 ,  13.887092 ,  -7.191962 ,   2.8393643],
#        dtype=float32),
#  'probabilities': array([1.9811394e-10, 3.4410416e-08, 8.7829823e-09, 4.5472443e-05,
#         2.8776778e-11, 1.3607721e-10, 1.2760607e-15, 9.9993849e-01,
#         7.0057804e-10, 1.5922315e-05], dtype=float32)}


#%% Training a DNN Using Plain TensorFlow
# The first step is the construction
# phase, building the TensorFlow graph. The second step is the execution
# phase, where you actually run the graph to train the model.

# Construction Phase
n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

# Let’s use it to create
# the deep neural network! The first hidden layer takes X as its input. The second takes
# the output of the first hidden layer as its input. And finally, the output layer takes the
# output of the second hidden layer as its input.

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)

# we will use cross entropy
# We can then use TensorFlow’s reduce_mean()
# function to compute the mean cross entropy over all instances.
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# now we need to
# define a GradientDescentOptimizer that will tweak the model parameters to minimize
# the cost function
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# The last important step in the construction phase is to specify how to evaluate the
# model. We will simply use accuracy as our performance measure.
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Execution Phase
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_epochs = 40
batch_size = 50

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# at the end of each epoch, the code evaluates the model on the last mini-batch and on
# the full training set, and it prints out the result
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

    save_path = saver.save(sess, "./my_model_final.ckpt")

# 0 Batch accuracy: 0.9 Validation accuracy: 0.9024
# 1 Batch accuracy: 0.92 Validation accuracy: 0.9254
# 2 Batch accuracy: 0.94 Validation accuracy: 0.9372
# 3 Batch accuracy: 0.9 Validation accuracy: 0.9416
# 4 Batch accuracy: 0.94 Validation accuracy: 0.9472
# 5 Batch accuracy: 0.94 Validation accuracy: 0.9512
# 6 Batch accuracy: 1.0 Validation accuracy: 0.9548
# 7 Batch accuracy: 0.94 Validation accuracy: 0.9612
# 8 Batch accuracy: 0.96 Validation accuracy: 0.962
# 9 Batch accuracy: 0.94 Validation accuracy: 0.9648
# 10 Batch accuracy: 0.92 Validation accuracy: 0.9652
# 11 Batch accuracy: 0.98 Validation accuracy: 0.9668
# 12 Batch accuracy: 0.98 Validation accuracy: 0.9686
# 13 Batch accuracy: 0.98 Validation accuracy: 0.97
# 14 Batch accuracy: 1.0 Validation accuracy: 0.9696
# 15 Batch accuracy: 0.94 Validation accuracy: 0.9718
# 16 Batch accuracy: 0.98 Validation accuracy: 0.973
# 17 Batch accuracy: 1.0 Validation accuracy: 0.9728
# 18 Batch accuracy: 0.98 Validation accuracy: 0.9748
# 19 Batch accuracy: 0.96 Validation accuracy: 0.9752
# 20 Batch accuracy: 1.0 Validation accuracy: 0.975
# 21 Batch accuracy: 1.0 Validation accuracy: 0.9732
# 22 Batch accuracy: 0.96 Validation accuracy: 0.9752
# 23 Batch accuracy: 0.98 Validation accuracy: 0.9766
# 24 Batch accuracy: 0.98 Validation accuracy: 0.9758
# 25 Batch accuracy: 1.0 Validation accuracy: 0.9762
# 26 Batch accuracy: 0.92 Validation accuracy: 0.9768
# 27 Batch accuracy: 1.0 Validation accuracy: 0.9774
# 28 Batch accuracy: 0.94 Validation accuracy: 0.9782
# 29 Batch accuracy: 0.98 Validation accuracy: 0.9776
# 30 Batch accuracy: 1.0 Validation accuracy: 0.9774
# 31 Batch accuracy: 1.0 Validation accuracy: 0.9778
# 32 Batch accuracy: 0.96 Validation accuracy: 0.9778
# 33 Batch accuracy: 0.98 Validation accuracy: 0.9784
# 34 Batch accuracy: 0.98 Validation accuracy: 0.9784
# 35 Batch accuracy: 1.0 Validation accuracy: 0.9778
# 36 Batch accuracy: 1.0 Validation accuracy: 0.9792
# 37 Batch accuracy: 1.0 Validation accuracy: 0.9784
# 38 Batch accuracy: 0.98 Validation accuracy: 0.9796
# 39 Batch accuracy: 1.0 Validation accuracy: 0.9782


# Using the Neural Network
# you would need to apply
# the softmax() function to the logits, but if you just want to predict a class, you can
# simply pick the class that has the highest logit value (using the argmax() function
# does the trick)

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt") # or better, use save_path
    X_new_scaled = X_test[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

y_test[:20]
# Out[17]: array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4])
y_pred
# Out[18]: array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4], dtype=int64)

