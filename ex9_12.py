# Implement Logistic Regression with Mini-batch Gradient Descent using Tensor‐
# Flow. Train it and evaluate it on the moons dataset. Try
# adding all the bells and whistles:
# • Define the graph within a logistic_regression() function that can be reused
# easily.
# • Save checkpoints using a Saver at regular intervals during training, and save
# the final model at the end of training.
# • Restore the last checkpoint upon startup if training was interrupted.
# • Define the graph using nice scopes so the graph looks good in TensorBoard.
# • Add summaries to visualize the learning curves in TensorBoard.
# • Try tweaking some hyperparameters such as the learning rate or the minibatch
# size and look at the shape of the learning curve.

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# First, let's create the moons dataset using Scikit-Learn's make_moons() function:
from sklearn.datasets import make_moons

m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

# Let's take a peek at the dataset:
plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label="Positive")
plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label="Negative")
plt.legend()
plt.show()

# We must not forget to add an extra bias feature (x0=1) to every instance.
# For this, we just need to add a column full of 1s on the left of the input matrix X:
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]

# Let's check:
X_moons_with_bias[:5]
# Out[5]:
# array([[ 1.        , -0.05146968,  0.44419863],
#        [ 1.        ,  1.03201691, -0.41974116],
#        [ 1.        ,  0.86789186, -0.25482711],
#        [ 1.        ,  0.288851  , -0.44866862],
#        [ 1.        , -0.83343911,  0.53505665]])

# Looks good. Now let's reshape y_train to make it a column vector
# (i.e. a 2D array with a single column):
y_moons_column_vector = y_moons.reshape(-1, 1)

# Now let's split the data into a training set and a test set:
test_ratio = 0.2
test_size = int(m * test_ratio)

X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]

# Ok, now let's create a small function to generate training batches.
# In this implementation we will just pick random instances from the training set
# for each batch. This means that a single batch may contain the same instance multiple times,
# and also a single epoch may not cover all the training instances
# (in fact it will generally cover only about two thirds of the instances).
# However, in practice this is not an issue and it simplifies the code:

def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

# Let's look at a small batch:
X_batch, y_batch = random_batch(X_train, y_train, 5)
X_batch
# Out[11]:
# array([[1.        , 1.09928901, 0.02042887],
#        [1.        , 1.18922384, 0.29023942],
#        [1.        , 0.40418029, 0.85682111],
#        [1.        , 0.70243601, 0.65930637],
#        [1.        , 1.87236982, 0.16767881]])

y_batch
# Out[12]:
# array([[0],
#        [0],
#        [0],
#        [0],
#        [1]], dtype=int64)

# Great! Now that the data is ready to be fed to the model, we need to build that model.
# Let's start with a simple implementation, then we will add all the bells and whistles.
# First let's reset the default graph.
reset_graph()

# The moons dataset has two input features,
# since each instance is a point on a plane (i.e., 2-Dimensional):
n_inputs = 2

# Now let's build the Logistic Regression model.
X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed=42), name="theta")
logits = tf.matmul(X, theta, name="logits")
y_proba = 1 / (1 + tf.exp(-logits))

# In fact, TensorFlow has a nice function tf.sigmoid()
# that we can use to simplify the last line of the previous code:
y_proba = tf.sigmoid(logits)

# the log loss is a good cost function to use for Logistic Regression:
# One option is to implement it ourselves:
epsilon = 1e-7  # to avoid an overflow when computing the log
loss = -tf.reduce_mean(y * tf.log(y_proba + epsilon) + (1 - y) * tf.log(1 - y_proba + epsilon))

# But we might as well use TensorFlow's tf.losses.log_loss() function:
loss = tf.losses.log_loss(y, y_proba)  # uses epsilon = 1e-7 by default

# The rest is pretty standard:
# let's create the optimizer and tell it to minimize the cost function:
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# All we need now (in this minimal version) is the variable initializer:
init = tf.global_variables_initializer()


# And we are ready to train the model and use it for predictions!
# There's really nothing special about this code, ' \
#      'it's virtually the same as the one we used earlier for Linear Regression:
n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        loss_val = loss.eval({X: X_test, y: y_test})
        if epoch % 100 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)

    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})

# Epoch: 0 	Loss: 0.79260236
# Epoch: 100 	Loss: 0.34346345
# Epoch: 200 	Loss: 0.30754045
# Epoch: 300 	Loss: 0.29288897
# Epoch: 400 	Loss: 0.28533578
# Epoch: 500 	Loss: 0.2804781
# Epoch: 600 	Loss: 0.2780829
# Epoch: 700 	Loss: 0.2761544
# Epoch: 800 	Loss: 0.27551997
# Epoch: 900 	Loss: 0.27491233

# For each instance in the test set,
# y_proba_val contains the estimated probability that it belongs to the positive class,
# according to the model. For example, here are the first 5 estimated probabilities:
y_proba_val[:5]
# Out[24]:
# array([[0.54895616],
#        [0.70724374],
#        [0.51900256],
#        [0.9911136 ],
#        [0.5085905 ]], dtype=float32)


# To classify each instance, we can go for maximum likelihood:
# classify as positive any instance whose estimated probability is greater or equal to 0.5:
y_pred = (y_proba_val >= 0.5)
y_pred[:5]
# Out[25]:
# array([[ True],
#        [ True],
#        [ True],
#        [ True],
#        [ True]])

# Let's compute the model's precision and recall:
from sklearn.metrics import precision_score, recall_score

precision_score(y_test, y_pred)
# Out[26]: 0.8627450980392157

recall_score(y_test, y_pred)
# Out[27]: 0.8888888888888888

# Let's plot these predictions to see what they look like:
y_pred_idx = y_pred.reshape(-1) # a 1D array rather than a column vector
plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label="Positive")
plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label="Negative")
plt.legend()
plt.show()

# Well, that looks pretty bad, doesn't it?
# But let's not forget that the Logistic Regression model has a linear decision boundary,
# so this is actually close to the best we can do with this model
# (unless we add more features, as we will show in a second).

# Now let's start over, but this time we will add all the bells and whistles,
# as listed in the exercise:

# Before we start, we will add 4 more features to the inputs:
# x1^2, x2^2, x1^3 and x2^3. This was not part of the exercise,
# but it will demonstrate how adding features can improve the model.
# We will do this manually, but you could also add them using sklearn.preprocessing.PolynomialFeatures.
X_train_enhanced = np.c_[X_train,
                         np.square(X_train[:, 1]),
                         np.square(X_train[:, 2]),
                         X_train[:, 1] ** 3,
                         X_train[:, 2] ** 3]
X_test_enhanced = np.c_[X_test,
                        np.square(X_test[:, 1]),
                        np.square(X_test[:, 2]),
                        X_test[:, 1] ** 3,
                        X_test[:, 2] ** 3]

X_train_enhanced[:5]
# Out[29]:
# array([[ 1.00000000e+00, -5.14696757e-02,  4.44198631e-01,
#          2.64912752e-03,  1.97312424e-01, -1.36349734e-04,
#          8.76459084e-02],
#        [ 1.00000000e+00,  1.03201691e+00, -4.19741157e-01,
#          1.06505890e+00,  1.76182639e-01,  1.09915879e+00,
#         -7.39511049e-02],
#        [ 1.00000000e+00,  8.67891864e-01, -2.54827114e-01,
#          7.53236288e-01,  6.49368582e-02,  6.53727646e-01,
#         -1.65476722e-02],
#        [ 1.00000000e+00,  2.88850997e-01, -4.48668621e-01,
#          8.34348982e-02,  2.01303531e-01,  2.41002535e-02,
#         -9.03185778e-02],
#        [ 1.00000000e+00, -8.33439108e-01,  5.35056649e-01,
#          6.94620746e-01,  2.86285618e-01, -5.78924095e-01,
#          1.53179024e-01]])

# Ok, next let's reset the default graph:
reset_graph()

# Now let's define the logistic_regression() function to create the graph.
# We will leave out the definition of the inputs X and the targets y.
# We could include them here, but leaving them out will make it easier
# to use this function in a wide range of use cases
# (e.g. perhaps we will want to add some preprocessing steps
# for the inputs before we feed them to the Logistic Regression model).
def logistic_regression(X, y, initializer=None, seed=42, learning_rate=0.01):
    n_inputs_including_bias = int(X.get_shape()[1])
    with tf.name_scope("logistic_regression"):
        with tf.name_scope("model"):
            if initializer is None:
                initializer = tf.random_uniform([n_inputs_including_bias, 1], -1.0, 1.0, seed=seed)
            theta = tf.Variable(initializer, name="theta")
            logits = tf.matmul(X, theta, name="logits")
            y_proba = tf.sigmoid(logits)

        with tf.name_scope("train"):
            loss = tf.losses.log_loss(y, y_proba, scope="loss")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)
            loss_summary = tf.summary.scalar('log_loss', loss)

        with tf.name_scope("init"):
            init = tf.global_variables_initializer()

        with tf.name_scope("save"):
            saver = tf.train.Saver()

    return y_proba, loss, training_op, loss_summary, init, saver

# Let's create a little function to get the name of the log directory
# to save the summaries for Tensorboard:

from datetime import datetime

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

# Next, let's create the graph, using the logistic_regression() function.
# We will also create the FileWriter to save the summaries to the log directory for Tensorboard:
n_inputs = 2 + 4
logdir = log_dir("logreg")

X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(X, y)

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# You can try interrupting training to verify
# that it does indeed restore the last checkpoint when you start it again.
n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = "/tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_logreg_model"

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        # if the checkpoint file exists, restore the model and load the epoch number
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test_enhanced, y: y_test})
        file_writer.add_summary(summary_str, epoch)
        if epoch % 500 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))

    saver.save(sess, final_model_path)
    y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
    os.remove(checkpoint_epoch_path)

# Epoch: 0 	Loss: 0.629985
# Epoch: 500 	Loss: 0.16122364
# Epoch: 1000 	Loss: 0.1190321
# Epoch: 1500 	Loss: 0.097329214
# Epoch: 2000 	Loss: 0.08369793
# Epoch: 2500 	Loss: 0.074375816
# Epoch: 3000 	Loss: 0.06750215
# Epoch: 3500 	Loss: 0.062206898
# Epoch: 4000 	Loss: 0.058026794
# Epoch: 4500 	Loss: 0.054562975
# Epoch: 5000 	Loss: 0.051708292
# Epoch: 5500 	Loss: 0.049237743
# Epoch: 6000 	Loss: 0.047167286
# Epoch: 6500 	Loss: 0.04537664
# Epoch: 7000 	Loss: 0.043818746
# Epoch: 7500 	Loss: 0.04237422
# Epoch: 8000 	Loss: 0.041089162
# Epoch: 8500 	Loss: 0.039970912
# Epoch: 9000 	Loss: 0.038920246
# Epoch: 9500 	Loss: 0.038010743
# Epoch: 10000 	Loss: 0.037155695

# Once again, we can make predictions by just classifying as positive
# all the instances whose estimated probability is greater or equal to 0.5:
y_pred = (y_proba_val >= 0.5)

precision_score(y_test, y_pred)
# Out[38]: 0.9797979797979798

recall_score(y_test, y_pred)
# Out[39]: 0.9797979797979798

y_pred_idx = y_pred.reshape(-1) # a 1D array rather than a column vector
plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label="Positive")
plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label="Negative")
plt.legend()
plt.show()

# Now that's much, much better! Apparently the new features really helped a lot.

# Try starting the tensorboard server, find the latest run and look at the learning curve
# (i.e., how the loss evaluated on the test set evolves as a function of the epoch number):


# Now you can play around with the hyperparameters
# (e.g. the batch_size or the learning_rate) and run training again and again,
# comparing the learning curves. You can even automate this process by implementing
# grid search or randomized search. Below is a simple implementation of a
# randomized search on both the batch size and the learning rate.
# For the sake of simplicity, the checkpoint mechanism was removed.
from scipy.stats import reciprocal

n_search_iterations = 10

for search_iteration in range(n_search_iterations):
    batch_size = np.random.randint(1, 100)
    learning_rate = reciprocal(0.0001, 0.1).rvs(random_state=search_iteration)

    n_inputs = 2 + 4
    logdir = log_dir("logreg")

    print("Iteration", search_iteration)
    print("  logdir:", logdir)
    print("  batch size:", batch_size)
    print("  learning_rate:", learning_rate)
    print("  training: ", end="")

    reset_graph()

    X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

    y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(
        X, y, learning_rate=learning_rate)

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    n_epochs = 10001
    n_batches = int(np.ceil(m / batch_size))

    final_model_path = "./my_logreg_model_%d" % search_iteration

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test_enhanced, y: y_test})
            file_writer.add_summary(summary_str, epoch)
            if epoch % 500 == 0:
                print(".", end="")

        saver.save(sess, final_model_path)

        print()
        y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
        y_pred = (y_proba_val >= 0.5)

        print("  precision:", precision_score(y_test, y_pred))
        print("  recall:", recall_score(y_test, y_pred))

# Iteration 0
#   logdir: tf_logs/logreg-run-20180808215000/
#   batch size: 54
#   learning_rate: 0.004430375245218265
#   training: .....................
#   precision: 0.9797979797979798
#   recall: 0.9797979797979798
# Iteration 1
#   logdir: tf_logs/logreg-run-20180808215332/
#   batch size: 22
#   learning_rate: 0.0017826497151386947
#   training: .....................
#   precision: 0.9797979797979798
#   recall: 0.9797979797979798
# Iteration 2
#   logdir: tf_logs/logreg-run-20180808220236/
#   batch size: 74
#   learning_rate: 0.00203228544324115
#   training: .....................
#   precision: 0.9696969696969697
#   recall: 0.9696969696969697
# Iteration 3
#   logdir: tf_logs/logreg-run-20180808220536/
#   batch size: 58
#   learning_rate: 0.004491523825137997
#   training: .....................
#   precision: 0.9797979797979798
#   recall: 0.9797979797979798
# Iteration 4
#   logdir: tf_logs/logreg-run-20180808220854/
#   batch size: 61
#   learning_rate: 0.07963234721775589
#   training: .....................
#   precision: 0.9801980198019802
#   recall: 1.0
# Iteration 5
#   logdir: tf_logs/logreg-run-20180808221156/
#   batch size: 92
#   learning_rate: 0.0004634250583294876
#   training: .....................
#   precision: 0.912621359223301
#   recall: 0.9494949494949495
# Iteration 6
#   logdir: tf_logs/logreg-run-20180808221404/
#   batch size: 74
#   learning_rate: 0.047706818419354494
#   training: .....................
#   precision: 0.98
#   recall: 0.98989898989899
# Iteration 7
#   logdir: tf_logs/logreg-run-20180808221646/
#   batch size: 58
#   learning_rate: 0.0001694044709524274
#   training: .....................
#   precision: 0.9
#   recall: 0.9090909090909091
# Iteration 8
#   logdir: tf_logs/logreg-run-20180808221955/
#   batch size: 61
#   learning_rate: 0.04171461199412461
#   training: .....................
#   precision: 0.9801980198019802
#   recall: 1.0
# Iteration 9
#   logdir: tf_logs/logreg-run-20180808222257/
#   batch size: 92
#   learning_rate: 0.00010742922968438615
#   training: .....................
#   precision: 0.8823529411764706
#   recall: 0.7575757575757576