import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

# matplotlib.rcParams['backend'] = 'Qt4Agg'
# matplotlib.rcParams['backend.qt5'] = 'PyQt4'

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# A couple utility functions to plot grayscale 28x28 image:

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()  # make the minimum == 0, so the padding looks white
    w,h = images.shape[1:]
    image = np.zeros(((w+pad)*n_rows+pad, (h+pad)*n_cols+pad))
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y*(h+pad)+pad):(y*(h+pad)+pad+h),(x*(w+pad)+pad):(x*(w+pad)+pad+w)] = images[y*n_cols+x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")


#%% PCA with a linear Autoencoder
# If the autoencoder uses only linear activations and the cost function is the Mean
# Squared Error (MSE), then it can be shown that it ends up performing Principal
# Component Analysis

# Build 3D dataset
import numpy.random as rnd

rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)

# Normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100])
X_test = scaler.transform(data[100:])

# Now let's build the Autoencoder...
# The following code builds a simple linear autoencoder to perform PCA on a 3D dataset,
# projecting it to 2D
reset_graph()

n_inputs = 3
n_hidden = 2  # codings
n_outputs = n_inputs

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden)
outputs = tf.layers.dense(hidden, n_outputs)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

# The two things to note are:
# • The number of outputs is equal to the number of inputs.
# • To perform simple PCA, we set activation_fn=None (i.e., all neurons are linear) and the cost function is the MSE.

n_iterations = 1000
codings = hidden  # the output of the hidden layer provides the codings

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X: X_train})   # no labels (unsupervised)
    codings_val = codings.eval(feed_dict={X: X_test})


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(9,4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_test[:,0], X_test[:,1], X_test[:,2])
# plt.show()

ax2 = fig.add_subplot(122)
ax2.plot(codings_val[:,0], codings_val[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.show()


#%% Stacked Autoencoders
# Let's use MNIST:
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

# Train all layers at once
# Let's build a stacked Autoencoder with 3 hidden layers and 1 output layer (ie. 2 stacked Autoencoders).
# We will use ELU activation, He initialization and L2 regularization.
# The code should look very familiar, except that there are no labels (no y):

reset_graph()

from functools import partial

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

he_init = tf.contrib.layers.variance_scaling_initializer() # He initialization
#Equivalent to:
#he_init = lambda shape, dtype=tf.float32: tf.truncated_normal(shape, 0., stddev=np.sqrt(2/shape[0]))
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.elu,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)

hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver() # not shown in the book

# Now let's train it! Note that we don't feed target values (y_batch is not used). This is unsupervised training.

n_epochs = 5
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            # print("\r{}%".format(100 * iteration // n_batches), end="") # not shown in the book
            sys.stdout.flush()                                          # not shown
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})   # not shown
        print("\r{}".format(epoch), "Train MSE:", loss_train)           # not shown
        saver.save(sess, "./my_model_all_layers.ckpt")                  # not shown

# 0 Train MSE: 0.020855438
# 1 Train MSE: 0.011372581
# 2 Train MSE: 0.010224564
# 3 Train MSE: 0.009900457
# 4 Train MSE: 0.010375758

# This function loads the model, evaluates it on the test set (it measures the reconstruction error),
# then it displays the original image and its reconstruction:
def show_reconstructed_digits(X, outputs, model_path = None, n_test_digits = 2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])

show_reconstructed_digits(X, outputs, "./my_model_all_layers.ckpt")


#%% Tying Weights
# It is common to tie the weights of the encoder and the decoder (weights_decoder = tf.transpose(weights_encoder)).
# Unfortunately this makes it impossible (or very tricky) to use the tf.layers.dense() function,
# so we need to build the Autoencoder manually:

reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0005

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.transpose(weights2, name="weights3")  # tied weights
weights4 = tf.transpose(weights1, name="weights4")  # tied weights

biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
outputs = tf.matmul(hidden3, weights4) + biases4

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_loss = regularizer(weights1) + regularizer(weights2)
loss = reconstruction_loss + reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_epochs = 5
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            # print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train MSE:", loss_train)
        saver.save(sess, "./my_model_tying_weights.ckpt")

# 0 Train MSE: 0.015602261
# 1 Train MSE: 0.016729439
# 2 Train MSE: 0.015856257
# 3 Train MSE: 0.017920662
# 4 Train MSE: 0.0178301

show_reconstructed_digits(X, outputs, "./my_model_tying_weights.ckpt")

# there are a few important things to note:
# • First, weight3 and weights4 are not variables, they are respectively the transpose
# of weights2 and weights1 (they are “tied” to them).
# • Second, since they are not variables, it’s no use regularizing them: we only regularize
# weights1 and weights2.
# • Third, biases are never tied, and never regularized.


#%% Training one Autoencoder at a time in multiple graphs
# There are many ways to train one Autoencoder at a time.
# The first approach is to train each Autoencoder using a different graph,
# then we create the Stacked Autoencoder by simply initializing it with the weights and biases copied from these Autoencoders.
# Let's create a function that will train one autoencoder and return the transformed training set
# (i.e., the output of the hidden layer) and the model parameters.

reset_graph()

from functools import partial

def train_autoencoder(X_train, n_neurons, n_epochs, batch_size,
                      learning_rate=0.01, l2_reg=0.0005, seed=42,
                      hidden_activation=tf.nn.elu,
                      output_activation=tf.nn.elu):
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(seed)

        n_inputs = X_train.shape[1]

        X = tf.placeholder(tf.float32, shape=[None, n_inputs])

        my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

        hidden = my_dense_layer(X, n_neurons, activation=hidden_activation, name="hidden")
        outputs = my_dense_layer(hidden, n_inputs, activation=output_activation, name="outputs")

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([reconstruction_loss] + reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = len(X_train) // batch_size
            for iteration in range(n_batches):
                # print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                indices = rnd.permutation(len(X_train))[:batch_size]
                X_batch = X_train[indices]
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
        params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        hidden_val = hidden.eval(feed_dict={X: X_train})
        return hidden_val, params["hidden/kernel:0"], params["hidden/bias:0"], params["outputs/kernel:0"], params[
            "outputs/bias:0"]

# Now let's train two Autoencoders. The first one is trained on the training data,
# and the second is trained on the previous Autoencoder's hidden layer output:

hidden_output, W1, b1, W4, b4 = train_autoencoder(mnist.train.images, n_neurons=300, n_epochs=4, batch_size=150,
                                                  output_activation=None)
_, W2, b2, W3, b3 = train_autoencoder(hidden_output, n_neurons=150, n_epochs=4, batch_size=150)

# 0 Train MSE: 0.01839451
# 1 Train MSE: 0.018561972
# 2 Train MSE: 0.019172184
# 3 Train MSE: 0.019527696
# 0 Train MSE: 0.004414982
# 1 Train MSE: 0.004626561
# 2 Train MSE: 0.004617909
# 3 Train MSE: 0.004764604

# Finally, we can create a Stacked Autoencoder
# by simply reusing the weights and biases from the Autoencoders we just trained:

reset_graph()

n_inputs = 28*28

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden1 = tf.nn.elu(tf.matmul(X, W1) + b1)
hidden2 = tf.nn.elu(tf.matmul(hidden1, W2) + b2)
hidden3 = tf.nn.elu(tf.matmul(hidden2, W3) + b3)
outputs = tf.matmul(hidden3, W4) + b4

show_reconstructed_digits(X, outputs)
plt.show()

# Training one Autoencoder at a time in a single graph
# Another approach is to use a single graph. To do this, we create the graph for the full Stacked Autoencoder,
# but then we also add operations to train each Autoencoder independently:
# phase 1 trains the bottom and top layer (ie. the first Autoencoder) and
# phase 2 trains the two middle layers (ie. the second Autoencoder).

reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])
weights3_init = initializer([n_hidden2, n_hidden3])
weights4_init = initializer([n_hidden3, n_outputs])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
weights4 = tf.Variable(weights4_init, dtype=tf.float32, name="weights4")

biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
outputs = tf.matmul(hidden3, weights4) + biases4

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate)

with tf.name_scope("phase1"):
    phase1_outputs = tf.matmul(hidden1, weights4) + biases4  # bypass hidden2 and hidden3
    phase1_reconstruction_loss = tf.reduce_mean(tf.square(phase1_outputs - X))
    phase1_reg_loss = regularizer(weights1) + regularizer(weights4)
    phase1_loss = phase1_reconstruction_loss + phase1_reg_loss
    phase1_training_op = optimizer.minimize(phase1_loss)

with tf.name_scope("phase2"):
    phase2_reconstruction_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))
    phase2_reg_loss = regularizer(weights2) + regularizer(weights3)
    phase2_loss = phase2_reconstruction_loss + phase2_reg_loss
    train_vars = [weights2, biases2, weights3, biases3]
    phase2_training_op = optimizer.minimize(phase2_loss, var_list=train_vars) # freeze hidden1

init = tf.global_variables_initializer()
saver = tf.train.Saver()

training_ops = [phase1_training_op, phase2_training_op]
reconstruction_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss]
n_epochs = [4, 4]
batch_sizes = [150, 150]

with tf.Session() as sess:
    init.run()
    for phase in range(2):
        print("Training phase #{}".format(phase + 1))
        for epoch in range(n_epochs[phase]):
            n_batches = mnist.train.num_examples // batch_sizes[phase]
            for iteration in range(n_batches):
                # print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                X_batch, y_batch = mnist.train.next_batch(batch_sizes[phase])
                sess.run(training_ops[phase], feed_dict={X: X_batch})
            loss_train = reconstruction_losses[phase].eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
            saver.save(sess, "./my_model_one_at_a_time.ckpt")
    loss_test = reconstruction_loss.eval(feed_dict={X: mnist.test.images})
    print("Test MSE:", loss_test)

# Training phase #1
# 0 Train MSE: 0.00792815
# 1 Train MSE: 0.0071743173
# 2 Train MSE: 0.008021195
# 3 Train MSE: 0.0073839147
# Training phase #2
# 0 Train MSE: 0.07069773
# 1 Train MSE: 0.0039470023
# 2 Train MSE: 0.0023026823
# 3 Train MSE: 0.0019282979
# Test MSE: 0.009706804

# Cache the frozen layer outputs
training_ops = [phase1_training_op, phase2_training_op]
reconstruction_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss]
n_epochs = [4, 4]
batch_sizes = [150, 150]

with tf.Session() as sess:
    init.run()
    for phase in range(2):
        print("Training phase #{}".format(phase + 1))
        if phase == 1:
            hidden1_cache = hidden1.eval(feed_dict={X: mnist.train.images})
        for epoch in range(n_epochs[phase]):
            n_batches = mnist.train.num_examples // batch_sizes[phase]
            for iteration in range(n_batches):
                # print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                if phase == 1:
                    indices = rnd.permutation(mnist.train.num_examples)
                    hidden1_batch = hidden1_cache[indices[:batch_sizes[phase]]]
                    feed_dict = {hidden1: hidden1_batch}
                    sess.run(training_ops[phase], feed_dict=feed_dict)
                else:
                    X_batch, y_batch = mnist.train.next_batch(batch_sizes[phase])
                    feed_dict = {X: X_batch}
                    sess.run(training_ops[phase], feed_dict=feed_dict)
            loss_train = reconstruction_losses[phase].eval(feed_dict=feed_dict)
            print("\r{}".format(epoch), "Train MSE:", loss_train)
            saver.save(sess, "./my_model_cache_frozen.ckpt")
    loss_test = reconstruction_loss.eval(feed_dict={X: mnist.test.images})
    print("Test MSE:", loss_test)

# Training phase #1
# 0 Train MSE: 0.007781427
# 1 Train MSE: 0.0073880567
# 2 Train MSE: 0.007893159
# 3 Train MSE: 0.008172775
# Training phase #2
# 0 Train MSE: 0.17783311
# 1 Train MSE: 0.006159089
# 2 Train MSE: 0.0026217313
# 3 Train MSE: 0.0021637985
# Test MSE: 0.009691488

# Visualizing the Reconstructions
n_test_digits = 2
X_test = mnist.test.images[:n_test_digits]

with tf.Session() as sess:
    saver.restore(sess, "./my_model_one_at_a_time.ckpt") # not shown in the book
    outputs_val = outputs.eval(feed_dict={X: X_test})

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

for digit_index in range(n_test_digits):
    plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
    plot_image(X_test[digit_index])
    plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
    plot_image(outputs_val[digit_index])

plt.show()

# Visualizing the extracted features
# For each neuron in the first hidden layer, you can create
# an image where a pixel’s intensity corresponds to the weight of the connection to
# the given neuron. For example, the following code plots the features learned by five
# neurons in the first hidden layer:

with tf.Session() as sess:
    saver.restore(sess, "./my_model_one_at_a_time.ckpt") # not shown in the book
    weights1_val = weights1.eval()

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plot_image(weights1_val.T[i])

plt.show()


#%% Unsupervised Pretraining Using Stacked Autoencoders
# Let's create a small neural network for MNIST classification:

reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_outputs = 10

learning_rate = 0.01
l2_reg = 0.0005

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
y = tf.placeholder(tf.int32, shape=[None])

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])
weights3_init = initializer([n_hidden2, n_outputs])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")

biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 = tf.Variable(tf.zeros(n_outputs), name="biases3")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
logits = tf.matmul(hidden2, weights3) + biases3

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
reg_loss = regularizer(weights1) + regularizer(weights2) + regularizer(weights3)
loss = cross_entropy + reg_loss
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
pretrain_saver = tf.train.Saver([weights1, weights2, biases1, biases2])
saver = tf.train.Saver()

# Regular training (without pretraining):

n_epochs = 4
batch_size = 150
n_labeled_instances = 20000

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = n_labeled_instances // batch_size
        for iteration in range(n_batches):
            # print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            indices = rnd.permutation(n_labeled_instances)[:batch_size]
            X_batch, y_batch = mnist.train.images[indices], mnist.train.labels[indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("\r{}".format(epoch), "Train accuracy:", accuracy_val, end=" ")
        saver.save(sess, "./my_model_supervised.ckpt")
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print("Test accuracy:", accuracy_val)

# 0 Train accuracy: 0.9533333 Test accuracy: 0.9356
# 1 Train accuracy: 0.94666666 Test accuracy: 0.9328
# 2 Train accuracy: 0.94666666 Test accuracy: 0.9361
# 3 Train accuracy: 0.97333336 Test accuracy: 0.951

# Now reusing the first two layers of the autoencoder we pretrained:

n_epochs = 4
batch_size = 150
n_labeled_instances = 20000

#training_op = optimizer.minimize(loss, var_list=[weights3, biases3])  # Freeze layers 1 and 2 (optional)

with tf.Session() as sess:
    init.run()
    pretrain_saver.restore(sess, "./my_model_cache_frozen.ckpt")
    for epoch in range(n_epochs):
        n_batches = n_labeled_instances // batch_size
        for iteration in range(n_batches):
            # print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            indices = rnd.permutation(n_labeled_instances)[:batch_size]
            X_batch, y_batch = mnist.train.images[indices], mnist.train.labels[indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("\r{}".format(epoch), "Train accuracy:", accuracy_val, end="\t")
        saver.save(sess, "./my_model_supervised_pretrained.ckpt")
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print("Test accuracy:", accuracy_val)

# 0 Train accuracy: 0.96	Test accuracy: 0.9265
# 1 Train accuracy: 0.96	Test accuracy: 0.9409
# 2 Train accuracy: 0.96666664	Test accuracy: 0.94
# 3 Train accuracy: 0.9866667	Test accuracy: 0.9394


#%% Denoising Autoencoders
# Another way to force the autoencoder to learn useful features is to add noise to its
# inputs, training it to recover the original, noise-free inputs. This prevents the autoencoder
# from trivially copying its inputs to its outputs, so it ends up having to find patterns
# in the data.

# Using Gaussian noise:
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01

noise_level = 1.0

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
X_noisy = X + noise_level * tf.random_normal(tf.shape(X))

hidden1 = tf.layers.dense(X_noisy, n_hidden1, activation=tf.nn.relu,
                          name="hidden1")
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, # not shown in the book
                          name="hidden2")                            # not shown
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, # not shown
                          name="hidden3")                            # not shown
outputs = tf.layers.dense(hidden3, n_outputs, name="outputs")        # not shown

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 10
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            # print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train MSE:", loss_train)
        saver.save(sess, "./my_model_stacked_denoising_gaussian.ckpt")

# 0 Train MSE: 0.044504177
# 1 Train MSE: 0.041691612
# 2 Train MSE: 0.040759146
# 3 Train MSE: 0.04183143
# 4 Train MSE: 0.039888136
# 5 Train MSE: 0.041126635
# 6 Train MSE: 0.041734286
# 7 Train MSE: 0.038011137
# 8 Train MSE: 0.042902924
# 9 Train MSE: 0.04103713


# Using dropout:
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01

dropout_rate = 0.3

training = tf.placeholder_with_default(False, shape=(), name='training')

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,
                          name="hidden1")
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, # not shown in the book
                          name="hidden2")                            # not shown
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, # not shown
                          name="hidden3")                            # not shown
outputs = tf.layers.dense(hidden3, n_outputs, name="outputs")        # not shown

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 10
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            # print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, training: True})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train MSE:", loss_train)
        saver.save(sess, "./my_model_stacked_denoising_dropout.ckpt")

# 0 Train MSE: 0.028475668
# 1 Train MSE: 0.029702863
# 2 Train MSE: 0.026610456
# 3 Train MSE: 0.025824282
# 4 Train MSE: 0.025248837
# 5 Train MSE: 0.025599884
# 6 Train MSE: 0.02500443
# 7 Train MSE: 0.024997592
# 8 Train MSE: 0.025508722
# 9 Train MSE: 0.027677704

show_reconstructed_digits(X, outputs, "./my_model_stacked_denoising_dropout.ckpt")
plt.show()


#%% Sparse Autoencoder
# by adding an appropriate term to the cost function, the autoencoder is pushed to reduce
# the number of active neurons in the coding layer.

p = 0.1
q = np.linspace(0.001, 0.999, 500)
kl_div = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
mse = (p - q)**2
plt.plot([p, p], [0, 0.3], "k:")
plt.text(0.05, 0.32, "Target\nsparsity", fontsize=14)
plt.plot(q, kl_div, "b-", label="KL divergence")
plt.plot(q, mse, "r--", label="MSE")
plt.legend(loc="upper left")
plt.xlabel("Actual sparsity")
plt.ylabel("Cost", rotation=0)
plt.axis([0, 1, 0, 0.95])
# plt.show()


reset_graph()

n_inputs = 28 * 28
n_hidden1 = 1000  # sparse codings
n_outputs = n_inputs

def kl_divergence(p, q):
    # Kullback Leibler divergence
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))

learning_rate = 0.01
sparsity_target = 0.1
sparsity_weight = 0.2

X = tf.placeholder(tf.float32, shape=[None, n_inputs])            # not shown in the book

hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.sigmoid) # not shown
outputs = tf.layers.dense(hidden1, n_outputs)                     # not shown

hidden1_mean = tf.reduce_mean(hidden1, axis=0) # batch mean
sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, hidden1_mean))
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE
loss = reconstruction_loss + sparsity_weight * sparsity_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 100
batch_size = 1000

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            # print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
        reconstruction_loss_val, sparsity_loss_val, loss_val = sess.run([reconstruction_loss, sparsity_loss, loss], feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train MSE:", reconstruction_loss_val, "\tSparsity loss:", sparsity_loss_val, "\tTotal loss:", loss_val)
        saver.save(sess, "./my_model_sparse.ckpt")

# 0 Train MSE: 0.13819282 	Sparsity loss: 0.4184436 	Total loss: 0.22188154
# 1 Train MSE: 0.059175685 	Sparsity loss: 0.0107475 	Total loss: 0.061325185
# 2 Train MSE: 0.05391724 	Sparsity loss: 0.019957596 	Total loss: 0.05790876
# 3 Train MSE: 0.04770173 	Sparsity loss: 0.03947288 	Total loss: 0.055596307
# 4 Train MSE: 0.04483115 	Sparsity loss: 0.011592638 	Total loss: 0.047149677
# 5 Train MSE: 0.040433716 	Sparsity loss: 0.09246861 	Total loss: 0.05892744
# 6 Train MSE: 0.038858183 	Sparsity loss: 0.045811564 	Total loss: 0.048020497
# 7 Train MSE: 0.03786088 	Sparsity loss: 0.07498159 	Total loss: 0.0528572
# 8 Train MSE: 0.033115048 	Sparsity loss: 0.020323787 	Total loss: 0.037179805
# 9 Train MSE: 0.031442735 	Sparsity loss: 0.09567076 	Total loss: 0.050576888
# 10 Train MSE: 0.02739209 	Sparsity loss: 0.06570464 	Total loss: 0.040533017
# 11 Train MSE: 0.02472922 	Sparsity loss: 0.088890865 	Total loss: 0.042507395
# 12 Train MSE: 0.02336216 	Sparsity loss: 0.05766206 	Total loss: 0.03489457
# 13 Train MSE: 0.023003293 	Sparsity loss: 0.061935123 	Total loss: 0.035390317
# 14 Train MSE: 0.021229863 	Sparsity loss: 0.025880724 	Total loss: 0.026406009
# 15 Train MSE: 0.02206253 	Sparsity loss: 0.48820597 	Total loss: 0.119703725
# 16 Train MSE: 0.01909861 	Sparsity loss: 0.032202616 	Total loss: 0.025539134
# 17 Train MSE: 0.018978441 	Sparsity loss: 0.12995227 	Total loss: 0.044968896
# 18 Train MSE: 0.01748447 	Sparsity loss: 0.035003617 	Total loss: 0.024485193
# 19 Train MSE: 0.017774194 	Sparsity loss: 0.110244006 	Total loss: 0.039822996
# 20 Train MSE: 0.016907932 	Sparsity loss: 0.03032121 	Total loss: 0.022972174
# 21 Train MSE: 0.01867781 	Sparsity loss: 0.36376804 	Total loss: 0.09143142
# 22 Train MSE: 0.01628339 	Sparsity loss: 0.055483416 	Total loss: 0.027380072
# 23 Train MSE: 0.015947493 	Sparsity loss: 0.07936464 	Total loss: 0.031820424
# 24 Train MSE: 0.015669184 	Sparsity loss: 0.14289424 	Total loss: 0.04424803
# 25 Train MSE: 0.014609116 	Sparsity loss: 0.1333071 	Total loss: 0.041270535
# 26 Train MSE: 0.014521924 	Sparsity loss: 0.109481774 	Total loss: 0.036418278
# 27 Train MSE: 0.013596003 	Sparsity loss: 0.07207918 	Total loss: 0.02801184
# 28 Train MSE: 0.014067101 	Sparsity loss: 0.15789701 	Total loss: 0.045646504
# 29 Train MSE: 0.013439132 	Sparsity loss: 0.14449537 	Total loss: 0.042338207
# 30 Train MSE: 0.013606219 	Sparsity loss: 0.18507265 	Total loss: 0.05062075
# 31 Train MSE: 0.0141945025 	Sparsity loss: 0.067886844 	Total loss: 0.027771872
# 32 Train MSE: 0.01375725 	Sparsity loss: 0.057575993 	Total loss: 0.02527245
# 33 Train MSE: 0.013665549 	Sparsity loss: 0.113637045 	Total loss: 0.036392957
# 34 Train MSE: 0.012644484 	Sparsity loss: 0.13469595 	Total loss: 0.03958367
# 35 Train MSE: 0.01287033 	Sparsity loss: 0.1685057 	Total loss: 0.04657147
# 36 Train MSE: 0.012667082 	Sparsity loss: 0.09313163 	Total loss: 0.031293407
# 37 Train MSE: 0.012425529 	Sparsity loss: 0.10538784 	Total loss: 0.033503097
# 38 Train MSE: 0.012333633 	Sparsity loss: 0.19235896 	Total loss: 0.050805427
# 39 Train MSE: 0.012335517 	Sparsity loss: 0.14231777 	Total loss: 0.040799074
# 40 Train MSE: 0.012621799 	Sparsity loss: 0.23267439 	Total loss: 0.059156675
# 41 Train MSE: 0.01238078 	Sparsity loss: 0.10489694 	Total loss: 0.03336017
# 42 Train MSE: 0.011778782 	Sparsity loss: 0.10161268 	Total loss: 0.03210132
# 43 Train MSE: 0.012041609 	Sparsity loss: 0.13594094 	Total loss: 0.039229795
# 44 Train MSE: 0.01161921 	Sparsity loss: 0.114726424 	Total loss: 0.034564495
# 45 Train MSE: 0.0116660595 	Sparsity loss: 0.15797785 	Total loss: 0.04326163
# 46 Train MSE: 0.011670612 	Sparsity loss: 0.13549832 	Total loss: 0.038770273
# 47 Train MSE: 0.011546319 	Sparsity loss: 0.15661722 	Total loss: 0.04286976
# 48 Train MSE: 0.011650643 	Sparsity loss: 0.4430765 	Total loss: 0.10026594
# 49 Train MSE: 0.011560009 	Sparsity loss: 0.17584758 	Total loss: 0.046729524
# 50 Train MSE: 0.011358404 	Sparsity loss: 0.188867 	Total loss: 0.049131803
# 51 Train MSE: 0.011540977 	Sparsity loss: 0.26338995 	Total loss: 0.06421897
# 52 Train MSE: 0.012060762 	Sparsity loss: 0.36756146 	Total loss: 0.085573055
# 53 Train MSE: 0.011235623 	Sparsity loss: 0.12488395 	Total loss: 0.036212415
# 54 Train MSE: 0.013885329 	Sparsity loss: 0.43312114 	Total loss: 0.100509554
# 55 Train MSE: 0.012711261 	Sparsity loss: 0.07718601 	Total loss: 0.028148465
# 56 Train MSE: 0.012189809 	Sparsity loss: 0.30150568 	Total loss: 0.072490945
# 57 Train MSE: 0.0129035795 	Sparsity loss: 0.24843478 	Total loss: 0.06259054
# 58 Train MSE: 0.024871243 	Sparsity loss: 0.23206005 	Total loss: 0.07128325
# 59 Train MSE: 0.013289707 	Sparsity loss: 0.19800198 	Total loss: 0.052890107
# 60 Train MSE: 0.017982066 	Sparsity loss: 0.1941407 	Total loss: 0.056810208
# 61 Train MSE: 0.012957121 	Sparsity loss: 0.22906558 	Total loss: 0.05877024
# 62 Train MSE: 0.014011876 	Sparsity loss: 0.47945678 	Total loss: 0.10990323
# 63 Train MSE: 0.0139767295 	Sparsity loss: 0.33368057 	Total loss: 0.08071285
# 64 Train MSE: 0.011993475 	Sparsity loss: 0.1780599 	Total loss: 0.04760546
# 65 Train MSE: 0.016537772 	Sparsity loss: 0.18629064 	Total loss: 0.053795904
# 66 Train MSE: 0.015783297 	Sparsity loss: 0.9842033 	Total loss: 0.21262395
# 67 Train MSE: 0.037234485 	Sparsity loss: 0.31194276 	Total loss: 0.09962304
# 68 Train MSE: 0.015956203 	Sparsity loss: 0.21817414 	Total loss: 0.059591033
# 69 Train MSE: 0.012510675 	Sparsity loss: 0.4404568 	Total loss: 0.10060204
# 70 Train MSE: 0.042280443 	Sparsity loss: 0.7450314 	Total loss: 0.19128674
# 71 Train MSE: 0.014856414 	Sparsity loss: 0.2390905 	Total loss: 0.062674515
# 72 Train MSE: 0.016518792 	Sparsity loss: 0.79720664 	Total loss: 0.17596012
# 73 Train MSE: 0.014023511 	Sparsity loss: 0.31666842 	Total loss: 0.077357195
# 74 Train MSE: 0.029884145 	Sparsity loss: 0.32379937 	Total loss: 0.09464402
# 75 Train MSE: 0.030682044 	Sparsity loss: 0.5052822 	Total loss: 0.1317385
# 76 Train MSE: 0.01906307 	Sparsity loss: 0.20066978 	Total loss: 0.059197027
# 77 Train MSE: 0.020837521 	Sparsity loss: 0.31109914 	Total loss: 0.08305735
# 78 Train MSE: 0.017557224 	Sparsity loss: 0.53586817 	Total loss: 0.12473086
# 79 Train MSE: 0.020547306 	Sparsity loss: 0.42234 	Total loss: 0.10501531
# 80 Train MSE: 0.016212685 	Sparsity loss: 0.11862521 	Total loss: 0.039937727
# 81 Train MSE: 0.012784495 	Sparsity loss: 0.87986225 	Total loss: 0.18875694
# 82 Train MSE: 0.014741396 	Sparsity loss: 2.1320314 	Total loss: 0.44114769
# 83 Train MSE: 0.014455906 	Sparsity loss: 0.66040385 	Total loss: 0.14653668
# 84 Train MSE: 0.014488939 	Sparsity loss: 0.27031606 	Total loss: 0.06855215
# 85 Train MSE: 0.026322221 	Sparsity loss: 0.9363368 	Total loss: 0.21358958
# 86 Train MSE: 0.015944703 	Sparsity loss: 0.6837334 	Total loss: 0.1526914
# 87 Train MSE: 0.012276509 	Sparsity loss: 1.4182136 	Total loss: 0.29591924
# 88 Train MSE: 0.019292591 	Sparsity loss: 0.8644198 	Total loss: 0.19217657
# 89 Train MSE: 0.03330836 	Sparsity loss: 0.29857114 	Total loss: 0.093022585
# 90 Train MSE: 0.031240847 	Sparsity loss: 0.4590028 	Total loss: 0.123041406
# 91 Train MSE: 0.014256637 	Sparsity loss: 0.10008387 	Total loss: 0.034273412
# 92 Train MSE: 0.013848867 	Sparsity loss: 0.38781145 	Total loss: 0.09141116
# 93 Train MSE: 0.016333997 	Sparsity loss: 0.16226617 	Total loss: 0.048787232
# 94 Train MSE: 0.012234485 	Sparsity loss: 0.11690221 	Total loss: 0.035614926
# 95 Train MSE: 0.014484293 	Sparsity loss: 0.10257237 	Total loss: 0.034998767
# 96 Train MSE: 0.018367378 	Sparsity loss: 0.23805305 	Total loss: 0.06597799
# 97 Train MSE: 0.013505243 	Sparsity loss: 0.10922566 	Total loss: 0.035350375
# 98 Train MSE: 0.013672282 	Sparsity loss: 0.08767074 	Total loss: 0.031206433
# 99 Train MSE: 0.022978393 	Sparsity loss: 0.33295625 	Total loss: 0.08956965

# Note that the coding layer must output values from 0 to 1, which is why we use the sigmoid activation function:
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.sigmoid)

# To speed up training, you can normalize the inputs between 0 and 1,
# and use the cross entropy instead of the MSE for the cost function:
logits = tf.layers.dense(hidden1, n_outputs)
outputs = tf.nn.sigmoid(logits)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
reconstruction_loss = tf.reduce_mean(xentropy)


#%% Variational Autoencoder
# instead of directly producing a coding for a given input, the
# encoder produces a mean coding μ and a standard deviation σ.

reset_graph()

from functools import partial

n_inputs = 28 * 28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20  # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.001

initializer = tf.contrib.layers.variance_scaling_initializer()

my_dense_layer = partial(
    tf.layers.dense,
    activation=tf.nn.elu,
    kernel_initializer=initializer)

X = tf.placeholder(tf.float32, [None, n_inputs])
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None)
hidden3_sigma = my_dense_layer(hidden2, n_hidden3, activation=None)
noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)
hidden3 = hidden3_mean + hidden3_sigma * noise
hidden4 = my_dense_layer(hidden3, n_hidden4)
hidden5 = my_dense_layer(hidden4, n_hidden5)
logits = my_dense_layer(hidden5, n_outputs, activation=None)
outputs = tf.sigmoid(logits)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
reconstruction_loss = tf.reduce_sum(xentropy)

eps = 1e-10 # smoothing term to avoid computing log(0) which is NaN
latent_loss = 0.5 * tf.reduce_sum(
    tf.square(hidden3_sigma) + tf.square(hidden3_mean)
    - 1 - tf.log(eps + tf.square(hidden3_sigma)))

loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 50
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            # print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss], feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)
        saver.save(sess, "./my_model_variational.ckpt")

# 0 Train total loss: 28261.855 	Reconstruction loss: 22455.545 	Latent loss: 5806.3105
# 1 Train total loss: 28730.49 	Reconstruction loss: 24579.414 	Latent loss: 4151.0757
# 2 Train total loss: 30567.258 	Reconstruction loss: 24246.602 	Latent loss: 6320.6553
# 3 Train total loss: 23649.092 	Reconstruction loss: 19839.688 	Latent loss: 3809.404
# 4 Train total loss: 22082.871 	Reconstruction loss: 19137.574 	Latent loss: 2945.2979
# 5 Train total loss: 27074.723 	Reconstruction loss: 20868.396 	Latent loss: 6206.327
# 6 Train total loss: 23357.857 	Reconstruction loss: 18172.258 	Latent loss: 5185.599
# 7 Train total loss: 20534.223 	Reconstruction loss: 17152.914 	Latent loss: 3381.3083
# 8 Train total loss: 18169.3 	Reconstruction loss: 15131.951 	Latent loss: 3037.3499
# 9 Train total loss: 17817.281 	Reconstruction loss: 14837.193 	Latent loss: 2980.0886
# 10 Train total loss: 17085.998 	Reconstruction loss: 14145.361 	Latent loss: 2940.6362
# 11 Train total loss: 16382.275 	Reconstruction loss: 13293.337 	Latent loss: 3088.938
# 12 Train total loss: 16756.027 	Reconstruction loss: 13734.796 	Latent loss: 3021.2317
# 13 Train total loss: 16325.559 	Reconstruction loss: 13077.0625 	Latent loss: 3248.4963
# 14 Train total loss: 17581.52 	Reconstruction loss: 14078.508 	Latent loss: 3503.012
# 15 Train total loss: 16614.982 	Reconstruction loss: 13412.615 	Latent loss: 3202.367
# 16 Train total loss: 15546.8955 	Reconstruction loss: 12278.746 	Latent loss: 3268.1494
# 17 Train total loss: 16573.285 	Reconstruction loss: 13177.893 	Latent loss: 3395.3916
# 18 Train total loss: 16481.506 	Reconstruction loss: 13047.48 	Latent loss: 3434.0256
# 19 Train total loss: 16430.312 	Reconstruction loss: 12963.243 	Latent loss: 3467.0703
# 20 Train total loss: 19097.97 	Reconstruction loss: 15518.897 	Latent loss: 3579.0728
# 21 Train total loss: 24366.5 	Reconstruction loss: 19327.387 	Latent loss: 5039.113
# 22 Train total loss: 30578.05 	Reconstruction loss: 24170.74 	Latent loss: 6407.3105
# 23 Train total loss: 27188.469 	Reconstruction loss: 21111.084 	Latent loss: 6077.385
# 24 Train total loss: 25659.05 	Reconstruction loss: 20664.066 	Latent loss: 4994.9854
# 25 Train total loss: 23967.191 	Reconstruction loss: 20520.281 	Latent loss: 3446.9102
# 26 Train total loss: 19484.469 	Reconstruction loss: 15894.319 	Latent loss: 3590.15
# 27 Train total loss: 18687.521 	Reconstruction loss: 15196.082 	Latent loss: 3491.4392
# 28 Train total loss: 16711.973 	Reconstruction loss: 13566.2705 	Latent loss: 3145.7024
# 29 Train total loss: 16341.48 	Reconstruction loss: 13081.887 	Latent loss: 3259.5933
# 30 Train total loss: 22157.67 	Reconstruction loss: 18052.79 	Latent loss: 4104.881
# 31 Train total loss: 15743.012 	Reconstruction loss: 12551.7705 	Latent loss: 3191.2417
# 32 Train total loss: 15725.622 	Reconstruction loss: 12501.877 	Latent loss: 3223.7454
# 33 Train total loss: 15862.157 	Reconstruction loss: 12547.148 	Latent loss: 3315.0088
# 34 Train total loss: 20555.512 	Reconstruction loss: 16999.752 	Latent loss: 3555.7595
# 35 Train total loss: 19468.027 	Reconstruction loss: 15880.239 	Latent loss: 3587.7876
# 36 Train total loss: 25161.148 	Reconstruction loss: 20782.352 	Latent loss: 4378.798
# 37 Train total loss: 29715.254 	Reconstruction loss: 22401.691 	Latent loss: 7313.5635
# 38 Train total loss: 29301.363 	Reconstruction loss: 20735.906 	Latent loss: 8565.457
# 39 Train total loss: 27920.21 	Reconstruction loss: 20246.713 	Latent loss: 7673.498
# 40 Train total loss: 26252.258 	Reconstruction loss: 20374.021 	Latent loss: 5878.2363
# 41 Train total loss: 18461.926 	Reconstruction loss: 15410.015 	Latent loss: 3051.9106
# 42 Train total loss: 17515.07 	Reconstruction loss: 14325.928 	Latent loss: 3189.1436
# 43 Train total loss: 16553.39 	Reconstruction loss: 13265.132 	Latent loss: 3288.2595
# 44 Train total loss: 16231.988 	Reconstruction loss: 12967.317 	Latent loss: 3264.6707
# 45 Train total loss: 16120.725 	Reconstruction loss: 12738.108 	Latent loss: 3382.6157
# 46 Train total loss: 16154.564 	Reconstruction loss: 12852.659 	Latent loss: 3301.9058
# 47 Train total loss: 15490.008 	Reconstruction loss: 12050.996 	Latent loss: 3439.012
# 48 Train total loss: 15644.622 	Reconstruction loss: 12321.411 	Latent loss: 3323.2107
# 49 Train total loss: 15344.52 	Reconstruction loss: 11879.721 	Latent loss: 3464.7988


reset_graph()

from functools import partial

n_inputs = 28 * 28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20  # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.001

initializer = tf.contrib.layers.variance_scaling_initializer()
my_dense_layer = partial(
    tf.layers.dense,
    activation=tf.nn.elu,
    kernel_initializer=initializer)

X = tf.placeholder(tf.float32, [None, n_inputs])
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None)
hidden3_gamma = my_dense_layer(hidden2, n_hidden3, activation=None)
noise = tf.random_normal(tf.shape(hidden3_gamma), dtype=tf.float32)
hidden3 = hidden3_mean + tf.exp(0.5 * hidden3_gamma) * noise
hidden4 = my_dense_layer(hidden3, n_hidden4)
hidden5 = my_dense_layer(hidden4, n_hidden5)
logits = my_dense_layer(hidden5, n_outputs, activation=None)
outputs = tf.sigmoid(logits)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
reconstruction_loss = tf.reduce_sum(xentropy)
latent_loss = 0.5 * tf.reduce_sum(
    tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)
loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# Generating Digits
# Let's train the model and generate a few random digits:

n_digits = 60
n_epochs = 50
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            # print("\r{}%".format(100 * iteration // n_batches), end="")  # not shown in the book
            sys.stdout.flush()  # not shown
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss],
                                                                      feed_dict={X: X_batch})  # not shown
        print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val,
              "\tLatent loss:", latent_loss_val)  # not shown
        saver.save(sess, "./my_model_variational.ckpt")  # not shown

    codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})

# 0 Train total loss: 33315.27 	Reconstruction loss: 25009.455 	Latent loss: 8305.814
# 1 Train total loss: 33646.273 	Reconstruction loss: 24286.951 	Latent loss: 9359.322
# 2 Train total loss: 27939.764 	Reconstruction loss: 24002.742 	Latent loss: 3937.0217
# 3 Train total loss: 22096.383 	Reconstruction loss: 19474.904 	Latent loss: 2621.479
# 4 Train total loss: 22511.328 	Reconstruction loss: 18936.871 	Latent loss: 3574.4565
# 5 Train total loss: 18572.629 	Reconstruction loss: 15839.473 	Latent loss: 2733.157
# 6 Train total loss: 18407.855 	Reconstruction loss: 15571.253 	Latent loss: 2836.603
# 7 Train total loss: 17591.639 	Reconstruction loss: 14627.58 	Latent loss: 2964.059
# 8 Train total loss: 17182.797 	Reconstruction loss: 14191.769 	Latent loss: 2991.029
# 9 Train total loss: 16638.107 	Reconstruction loss: 13663.88 	Latent loss: 2974.2275
# 10 Train total loss: 21527.021 	Reconstruction loss: 17843.12 	Latent loss: 3683.9028
# 11 Train total loss: 16875.22 	Reconstruction loss: 13753.284 	Latent loss: 3121.9365
# 12 Train total loss: 16529.004 	Reconstruction loss: 13283.264 	Latent loss: 3245.7402
# 13 Train total loss: 16115.262 	Reconstruction loss: 13111.149 	Latent loss: 3004.1128
# 14 Train total loss: 17034.402 	Reconstruction loss: 13504.899 	Latent loss: 3529.5034
# 15 Train total loss: 16867.605 	Reconstruction loss: 13790.87 	Latent loss: 3076.7354
# 16 Train total loss: 15984.011 	Reconstruction loss: 12653.977 	Latent loss: 3330.0342
# 17 Train total loss: 15912.229 	Reconstruction loss: 12752.97 	Latent loss: 3159.2593
# 18 Train total loss: 16256.547 	Reconstruction loss: 13090.681 	Latent loss: 3165.8667
# 19 Train total loss: 16253.633 	Reconstruction loss: 12898.98 	Latent loss: 3354.6519
# 20 Train total loss: 16208.088 	Reconstruction loss: 12919.3 	Latent loss: 3288.7878
# 21 Train total loss: 15768.232 	Reconstruction loss: 12316.76 	Latent loss: 3451.4727
# 22 Train total loss: 16561.748 	Reconstruction loss: 13150.77 	Latent loss: 3410.9788
# 23 Train total loss: 16524.434 	Reconstruction loss: 12819.99 	Latent loss: 3704.4429
# 24 Train total loss: 15391.113 	Reconstruction loss: 12044.91 	Latent loss: 3346.2031
# 25 Train total loss: 15286.908 	Reconstruction loss: 11766.244 	Latent loss: 3520.6646
# 26 Train total loss: 15569.604 	Reconstruction loss: 12057.863 	Latent loss: 3511.7402
# 27 Train total loss: 15465.405 	Reconstruction loss: 12132.059 	Latent loss: 3333.3467
# 28 Train total loss: 17467.22 	Reconstruction loss: 13940.306 	Latent loss: 3526.9155
# 29 Train total loss: 19344.672 	Reconstruction loss: 15540.685 	Latent loss: 3803.9863
# 30 Train total loss: 21264.668 	Reconstruction loss: 17697.992 	Latent loss: 3566.6763
# 31 Train total loss: 33538.027 	Reconstruction loss: 25342.248 	Latent loss: 8195.779
# 32 Train total loss: 29679.238 	Reconstruction loss: 24319.965 	Latent loss: 5359.2725
# 33 Train total loss: 21870.703 	Reconstruction loss: 17521.822 	Latent loss: 4348.882
# 34 Train total loss: 17091.938 	Reconstruction loss: 13852.754 	Latent loss: 3239.1826
# 35 Train total loss: 16135.118 	Reconstruction loss: 12868.777 	Latent loss: 3266.341
# 36 Train total loss: 15482.806 	Reconstruction loss: 12175.146 	Latent loss: 3307.6592
# 37 Train total loss: 16178.811 	Reconstruction loss: 12822.32 	Latent loss: 3356.4897
# 38 Train total loss: 15834.706 	Reconstruction loss: 12428.898 	Latent loss: 3405.8074
# 39 Train total loss: 15333.648 	Reconstruction loss: 12100.46 	Latent loss: 3233.189
# 40 Train total loss: 14969.053 	Reconstruction loss: 11617.77 	Latent loss: 3351.2827
# 41 Train total loss: 15481.818 	Reconstruction loss: 12098.227 	Latent loss: 3383.5918
# 42 Train total loss: 16268.17 	Reconstruction loss: 13036.891 	Latent loss: 3231.2798
# 43 Train total loss: 15923.278 	Reconstruction loss: 12545.138 	Latent loss: 3378.1406
# 44 Train total loss: 17107.275 	Reconstruction loss: 13903.115 	Latent loss: 3204.1597
# 45 Train total loss: 16080.832 	Reconstruction loss: 12755.219 	Latent loss: 3325.6128
# 46 Train total loss: 15490.6045 	Reconstruction loss: 12201.823 	Latent loss: 3288.7812
# 47 Train total loss: 16068.583 	Reconstruction loss: 12711.501 	Latent loss: 3357.082
# 48 Train total loss: 14928.303 	Reconstruction loss: 11515.498 	Latent loss: 3412.8042
# 49 Train total loss: 16678.766 	Reconstruction loss: 13173.25 	Latent loss: 3505.5146

plt.figure(figsize=(8,50)) # not shown in the book
for iteration in range(n_digits):
    plt.subplot(n_digits, 10, iteration + 1)
    plot_image(outputs_val[iteration])

n_rows = 6
n_cols = 10
plot_multiple_images(outputs_val.reshape(-1, 28, 28), n_rows, n_cols)
plt.show()


# Encode & Decode
# Encode:

n_digits = 3
X_test, y_test = mnist.test.next_batch(batch_size)
codings = hidden3

with tf.Session() as sess:
    saver.restore(sess, "./my_model_variational.ckpt")
    codings_val = codings.eval(feed_dict={X: X_test})

# Decode:

with tf.Session() as sess:
    saver.restore(sess, "./my_model_variational.ckpt")
    outputs_val = outputs.eval(feed_dict={codings: codings_val})

# Let's plot the reconstructions:

fig = plt.figure(figsize=(8, 2.5 * n_digits))
for iteration in range(n_digits):
    plt.subplot(n_digits, 2, 1 + 2 * iteration)
    plot_image(X_test[iteration])
    plt.subplot(n_digits, 2, 2 + 2 * iteration)
    plot_image(outputs_val[iteration])


# Interpolate digits

n_iterations = 3
n_digits = 6
codings_rnd = np.random.normal(size=[n_digits, n_hidden3])

with tf.Session() as sess:
    saver.restore(sess, "./my_model_variational.ckpt")
    target_codings = np.roll(codings_rnd, -1, axis=0)
    for iteration in range(n_iterations + 1):
        codings_interpolate = codings_rnd + (target_codings - codings_rnd) * iteration / n_iterations
        outputs_val = outputs.eval(feed_dict={codings: codings_interpolate})
        plt.figure(figsize=(11, 1.5*n_iterations))
        for digit_index in range(n_digits):
            plt.subplot(1, n_digits, digit_index + 1)
            plot_image(outputs_val[digit_index])
        plt.show()

