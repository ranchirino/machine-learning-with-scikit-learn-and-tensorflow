import numpy as np
import tensorflow as tf
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

import matplotlib
import matplotlib.pyplot as plt

#%% Xavier and He Initialization
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

he_init = tf.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                          kernel_initializer=he_init, name="hidden1")

#%% Nonsaturating Activation Functions

# Leaky ReLU
def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha*z, z)

z = np.linspace(-5, 5, 200)

plt.plot(z, leaky_relu(z, 0.05), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([0, 0], [-0.5, 4.2], 'k-')
plt.grid(True)
props = dict(facecolor='black', shrink=0.1)
plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
plt.title("Leaky ReLU activation function", fontsize=14)
plt.axis([-5, 5, -0.5, 4.2])
plt.show()

# Implementing Leaky ReLU in TensorFlow:

def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

# Let's train a neural network on MNIST using the Leaky ReLU. First let's create the graph:
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=leaky_relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Let's load the data:
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

n_epochs = 40
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 5 == 0:
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

    save_path = saver.save(sess, "./my_model_final.ckpt")

# 0 Batch accuracy: 0.86 Validation accuracy: 0.9044
# 5 Batch accuracy: 0.94 Validation accuracy: 0.9494
# 10 Batch accuracy: 0.92 Validation accuracy: 0.9656
# 15 Batch accuracy: 0.94 Validation accuracy: 0.9712
# 20 Batch accuracy: 1.0 Validation accuracy: 0.9764
# 25 Batch accuracy: 1.0 Validation accuracy: 0.9774
# 30 Batch accuracy: 0.98 Validation accuracy: 0.978
# 35 Batch accuracy: 1.0 Validation accuracy: 0.9786

# ELU
def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)

plt.plot(z, elu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1, -1], 'k--')
plt.plot([0, 0], [-2.2, 3.2], 'k-')
plt.grid(True)
plt.title(r"ELU activation function ($\alpha=1$)", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])
plt.show()

# Implementing ELU in TensorFlow is trivial,
# just specify the activation function when building each layer:

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name="hidden1")


# SELU (scaled exponential linear units)
# During training, a neural network composed of a stack of dense layers
# using the SELU activation function will self-normalize:
# the output of each layer will tend to preserve the same mean and variance during training,
# which solves the vanishing/exploding gradients problem.
# As a result, this activation function outperforms the other activation functions
# very significantly for such neural nets, so you should really try it out.

def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * elu(z, alpha)

# The tf.nn.selu() function was added in TensorFlow 1.4

plt.plot(z, selu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1.758, -1.758], 'k--')
plt.plot([0, 0], [-2.2, 3.2], 'k-')
plt.grid(True)
plt.title(r"SELU activation function", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])
plt.show()

# By default, the SELU hyperparameters (scale and alpha) are tuned in such a way
# that the mean remains close to 0, and the standard deviation remains close to 1
# (assuming the inputs are standardized with mean 0 and standard deviation 1 too).
# Using this activation function, even a 100 layer deep neural network preserves
# roughly mean 0 and standard deviation 1 across all layers,
# avoiding the exploding/vanishing gradients problem:

np.random.seed(42)
Z = np.random.normal(size=(500, 100))
for layer in range(100):
    W = np.random.normal(size=(100, 100), scale=np.sqrt(1/100))
    Z = selu(np.dot(Z, W))
    means = np.mean(Z, axis=1)
    stds = np.std(Z, axis=1)
    if layer % 10 == 0:
        print("Layer {}: {:.2f} < mean < {:.2f}, {:.2f} < std deviation < {:.2f}".format(
            layer, means.min(), means.max(), stds.min(), stds.max()))

# Layer 0: -0.26 < mean < 0.27, 0.74 < std deviation < 1.27
# Layer 10: -0.24 < mean < 0.27, 0.74 < std deviation < 1.27
# Layer 20: -0.17 < mean < 0.18, 0.74 < std deviation < 1.24
# Layer 30: -0.27 < mean < 0.24, 0.78 < std deviation < 1.20
# Layer 40: -0.38 < mean < 0.39, 0.74 < std deviation < 1.25
# Layer 50: -0.27 < mean < 0.31, 0.73 < std deviation < 1.27
# Layer 60: -0.26 < mean < 0.43, 0.74 < std deviation < 1.35
# Layer 70: -0.19 < mean < 0.21, 0.75 < std deviation < 1.21
# Layer 80: -0.18 < mean < 0.16, 0.72 < std deviation < 1.19
# Layer 90: -0.19 < mean < 0.16, 0.75 < std deviation < 1.20

# Let's create a neural net for MNIST using the SELU activation function:

def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))

reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=selu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=selu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 40
batch_size = 50

# Now let's train it. Do not forget to scale the inputs to mean 0 and standard deviation 1:

means = X_train.mean(axis=0, keepdims=True)
stds = X_train.std(axis=0, keepdims=True) + 1e-10
X_val_scaled = (X_valid - means) / stds

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch_scaled = (X_batch - means) / stds
            sess.run(training_op, feed_dict={X: X_batch_scaled, y: y_batch})
        if epoch % 5 == 0:
            acc_batch = accuracy.eval(feed_dict={X: X_batch_scaled, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_val_scaled, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

    save_path = saver.save(sess, "./my_model_final_selu.ckpt")

# 0 Batch accuracy: 0.88 Validation accuracy: 0.923
# 5 Batch accuracy: 0.98 Validation accuracy: 0.9576
# 10 Batch accuracy: 1.0 Validation accuracy: 0.9662
# 15 Batch accuracy: 0.96 Validation accuracy: 0.9682
# 20 Batch accuracy: 1.0 Validation accuracy: 0.9694
# 25 Batch accuracy: 1.0 Validation accuracy: 0.969
# 30 Batch accuracy: 1.0 Validation accuracy: 0.9694
# 35 Batch accuracy: 1.0 Validation accuracy: 0.97


#%% Batch Normalization
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

training = tf.placeholder_with_default(False, shape=(), name='training')

hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
bn1_act = tf.nn.elu(bn1)

hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
bn2_act = tf.nn.elu(bn2)

logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
logits = tf.layers.batch_normalization(logits_before_bn, training=training,
                                       momentum=0.9)

# To avoid repeating the same parameters over and over again,
# we can use Python's partial() function:
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
training = tf.placeholder_with_default(False, shape=(), name='training')

from functools import partial

my_batch_norm_layer = partial(tf.layers.batch_normalization,
                              training=training, momentum=0.9)

hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
bn1 = my_batch_norm_layer(hidden1)
bn1_act = tf.nn.elu(bn1)
hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
bn2 = my_batch_norm_layer(hidden2)
bn2_act = tf.nn.elu(bn2)
logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
logits = my_batch_norm_layer(logits_before_bn)

# Let's build a neural net for MNIST,
# using the ELU activation function and Batch Normalization at each layer:
reset_graph()

batch_norm_momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')

with tf.name_scope("dnn"):
    he_init = tf.variance_scaling_initializer()

    my_batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=training,
            momentum=batch_norm_momentum)

    my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=he_init)

    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
    hidden2 = my_dense_layer(bn1, n_hidden2, name="hidden2")
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
    logits_before_bn = my_dense_layer(bn2, n_outputs, name="outputs")
    logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run([training_op, extra_update_ops],
                     feed_dict={training: True, X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")

# 0 Validation accuracy: 0.9042
# 1 Validation accuracy: 0.928
# 2 Validation accuracy: 0.9374
# 3 Validation accuracy: 0.9474
# 4 Validation accuracy: 0.9532
# 5 Validation accuracy: 0.9572
# 6 Validation accuracy: 0.9626
# 7 Validation accuracy: 0.9628
# 8 Validation accuracy: 0.9664
# 9 Validation accuracy: 0.968
# 10 Validation accuracy: 0.9694
# 11 Validation accuracy: 0.9696
# 12 Validation accuracy: 0.971
# 13 Validation accuracy: 0.971
# 14 Validation accuracy: 0.9728
# 15 Validation accuracy: 0.9734
# 16 Validation accuracy: 0.9728
# 17 Validation accuracy: 0.975
# 18 Validation accuracy: 0.9752
# 19 Validation accuracy: 0.976

# Of course, if you train for longer it will get much better accuracy,
# but with such a shallow network, Batch Norm and ELU are unlikely to have very positive impact:
# they shine mostly for much deeper nets.

#%%Gradient Clipping
# Let's create a simple neural net for MNIST and add gradient clipping.
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
    logits = tf.layers.dense(hidden5, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

# Now we apply gradient clipping. For this, we need to get the gradients,
# use the clip_by_value() function to clip them, then apply them:
threshold = 1.0

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
              for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(capped_gvs)

# The rest is the same as usual:
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")

# 0 Validation accuracy: 0.2876
# 1 Validation accuracy: 0.794
# 2 Validation accuracy: 0.8798
# 3 Validation accuracy: 0.9056
# 4 Validation accuracy: 0.9162
# 5 Validation accuracy: 0.9218
# 6 Validation accuracy: 0.9292
# 7 Validation accuracy: 0.9356
# 8 Validation accuracy: 0.938
# 9 Validation accuracy: 0.9416
# 10 Validation accuracy: 0.9456
# 11 Validation accuracy: 0.947
# 12 Validation accuracy: 0.9476
# 13 Validation accuracy: 0.9532
# 14 Validation accuracy: 0.9564
# 15 Validation accuracy: 0.9566
# 16 Validation accuracy: 0.9576
# 17 Validation accuracy: 0.9588
# 18 Validation accuracy: 0.9622
# 19 Validation accuracy: 0.9612


#%% Reusing Pretrained Layers

# Reusing a TensorFlow Model
# First you need to load the graph's structure.
# The import_meta_graph() function does just that,
# loading the graph's operations into the default graph,
# and returning a Saver that you can then use to restore the model's state.
# Note that by default, a Saver saves the structure of the graph into a .meta file,
# so that's the file you should load:

reset_graph()
saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")

# Next you need to get a handle on all the operations you will need for training.
# If you don't know the graph's structure, you can list all the operations:
for op in tf.get_default_graph().get_operations():
    print(op.name)

# Once you know which operations you need,
# you can get a handle on them using the graph's get_operation_by_name()
# or get_tensor_by_name() methods:
X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")

training_op = tf.get_default_graph().get_operation_by_name("GradientDescent")

# If you are the author of the original model,
# you could make things easier for people who will reuse your model
# by giving operations very clear names and documenting them.
# Another approach is to create a collection containing all the important
# operations that people will want to get a handle on:
for op in (X, y, accuracy, training_op):
    tf.add_to_collection("my_important_ops", op)

# This way people who reuse your model will be able to simply write:
X, y, accuracy, training_op = tf.get_collection("my_important_ops")

# Now you can start a session, restore the model's state and continue training on your data:
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    # continue training the model...

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_new_model_final.ckpt")

# 0 Validation accuracy: 0.964
# 1 Validation accuracy: 0.9628
# 2 Validation accuracy: 0.9654
# 3 Validation accuracy: 0.9652
# 4 Validation accuracy: 0.9642
# 5 Validation accuracy: 0.965
# 6 Validation accuracy: 0.9688
# 7 Validation accuracy: 0.9688
# 8 Validation accuracy: 0.9684
# 9 Validation accuracy: 0.9688
# 10 Validation accuracy: 0.9706
# 11 Validation accuracy: 0.9716
# 12 Validation accuracy: 0.9672
# 13 Validation accuracy: 0.9698
# 14 Validation accuracy: 0.971
# 15 Validation accuracy: 0.9724
# 16 Validation accuracy: 0.972
# 17 Validation accuracy: 0.9712
# 18 Validation accuracy: 0.9712
# 19 Validation accuracy: 0.971

# In general you will want to reuse only the lower layers.
# If you are using import_meta_graph() it will load the whole graph,
# but you can simply ignore the parts you do not need. In this example,
# we add a new 4th hidden layer on top of the pretrained 3rd layer
# (ignoring the old 4th hidden layer). We also build a new output layer,
# the loss for this new output, and a new optimizer to minimize it.
# We also need another saver to save the whole graph
# (containing both the entire old graph plus the new operations),
# and an initialization operation to initialize all the new variables:
reset_graph()

n_hidden4 = 20  # new layer
n_outputs = 10  # new layer

saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

hidden3 = tf.get_default_graph().get_tensor_by_name("dnn/hidden4/Relu:0")

new_hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="new_hidden4")
new_logits = tf.layers.dense(new_hidden4, n_outputs, name="new_outputs")

with tf.name_scope("new_loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("new_eval"):
    correct = tf.nn.in_top_k(new_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("new_train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
new_saver = tf.train.Saver()

# And we can train this new model:

with tf.Session() as sess:
    init.run()
    saver.restore(sess, "./my_model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = new_saver.save(sess, "./my_new_model_final.ckpt")

# 0 Validation accuracy: 0.9244
# 1 Validation accuracy: 0.9452
# 2 Validation accuracy: 0.9534
# 3 Validation accuracy: 0.9582
# 4 Validation accuracy: 0.9606
# 5 Validation accuracy: 0.9566
# 6 Validation accuracy: 0.962
# 7 Validation accuracy: 0.9624
# 8 Validation accuracy: 0.964
# 9 Validation accuracy: 0.9646
# 10 Validation accuracy: 0.966
# 11 Validation accuracy: 0.966
# 12 Validation accuracy: 0.965
# 13 Validation accuracy: 0.9678
# 14 Validation accuracy: 0.9678
# 15 Validation accuracy: 0.969
# 16 Validation accuracy: 0.9692
# 17 Validation accuracy: 0.9704
# 18 Validation accuracy: 0.9684
# 19 Validation accuracy: 0.9674

# If you have access to the Python code that built the original graph,
# you can just reuse the parts you need and drop the rest:
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # reused
n_hidden2 = 50  # reused
n_hidden3 = 50  # reused
n_hidden4 = 20  # new!
n_outputs = 10  # new!

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")       # reused
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2") # reused
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3") # reused
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4") # new!
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")                         # new!

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# However, you must create one Saver to restore the pretrained model
# (giving it the list of variables to restore,
# or else it will complain that the graphs don't match),
# and another Saver to save the new model, once it is trained:
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="hidden[123]") # regular expression
restore_saver = tf.train.Saver(reuse_vars) # to restore layers 1-3

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")

    for epoch in range(n_epochs):                                            # not shown in the book
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size): # not shown
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})        # not shown
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})     # not shown
        print(epoch, "Validation accuracy:", accuracy_val)                   # not shown

    save_path = saver.save(sess, "./my_new_model_final.ckpt")

# 0 Validation accuracy: 0.9018
# 1 Validation accuracy: 0.9332
# 2 Validation accuracy: 0.9428
# 3 Validation accuracy: 0.947
# 4 Validation accuracy: 0.9516
# 5 Validation accuracy: 0.9528
# 6 Validation accuracy: 0.9556
# 7 Validation accuracy: 0.9592
# 8 Validation accuracy: 0.9586
# 9 Validation accuracy: 0.961
# 10 Validation accuracy: 0.9626
# 11 Validation accuracy: 0.9622
# 12 Validation accuracy: 0.9638
# 13 Validation accuracy: 0.9662
# 14 Validation accuracy: 0.9662
# 15 Validation accuracy: 0.9668
# 16 Validation accuracy: 0.9672
# 17 Validation accuracy: 0.9676
# 18 Validation accuracy: 0.968
# 19 Validation accuracy: 0.9676


# Reusing Models from Other Frameworks
# In this example, for each variable we want to reuse,
# we find its initializer's assignment operation, and we get its second input,
# which corresponds to the initialization value. When we run the initializer,
# we replace the initialization values with the ones we want, using a feed_dict:

reset_graph()

n_inputs = 2
n_hidden1 = 3

original_w = [[1., 2., 3.], [4., 5., 6.]] # Load the weights from the other framework
original_b = [7., 8., 9.]                 # Load the biases from the other framework

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
# [...] Build the rest of the model

# Get a handle on the assignment nodes for the hidden1 variables
graph = tf.get_default_graph()
assign_kernel = graph.get_operation_by_name("hidden1/kernel/Assign")
assign_bias = graph.get_operation_by_name("hidden1/bias/Assign")
init_kernel = assign_kernel.inputs[1]
init_bias = assign_bias.inputs[1]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init, feed_dict={init_kernel: original_w, init_bias: original_b})
    # [...] Train the model on your new task
    print(hidden1.eval(feed_dict={X: [[10.0, 11.0]]}))  # not shown in the book

# [[ 61.  83. 105.]]

# Note that we could also get a handle on the variables using get_collection()
# and specifying the scope:
tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden1")
# Out[51]:
# [<tf.Variable 'hidden1/kernel:0' shape=(2, 3) dtype=float32_ref>,
#  <tf.Variable 'hidden1/bias:0' shape=(3,) dtype=float32_ref>]

# Or we could use the graph's get_tensor_by_name() method:
tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
# Out[52]: <tf.Tensor 'hidden1/kernel:0' shape=(2, 3) dtype=float32_ref>

tf.get_default_graph().get_tensor_by_name("hidden1/bias:0")
# Out[53]: <tf.Tensor 'hidden1/bias:0' shape=(3,) dtype=float32_ref>


# Freezing the Lower Layers
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # reused
n_hidden2 = 50  # reused
n_hidden3 = 50  # reused
n_hidden4 = 20  # new!
n_outputs = 10  # new!

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")       # reused
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2") # reused
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3") # reused
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4") # new!
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")                         # new!

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):                                         # not shown in the book
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)     # not shown
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope="hidden[34]|outputs")
    training_op = optimizer.minimize(loss, var_list=train_vars)

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="hidden[123]") # regular expression
restore_saver = tf.train.Saver(reuse_vars) # to restore layers 1-3

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_new_model_final.ckpt")

# 0 Validation accuracy: 0.8952
# 1 Validation accuracy: 0.9302
# 2 Validation accuracy: 0.94
# 3 Validation accuracy: 0.9442
# 4 Validation accuracy: 0.9478
# 5 Validation accuracy: 0.9502
# 6 Validation accuracy: 0.9508
# 7 Validation accuracy: 0.9536
# 8 Validation accuracy: 0.9554
# 9 Validation accuracy: 0.9566
# 10 Validation accuracy: 0.956
# 11 Validation accuracy: 0.9566
# 12 Validation accuracy: 0.957
# 13 Validation accuracy: 0.9572
# 14 Validation accuracy: 0.9588
# 15 Validation accuracy: 0.9578
# 16 Validation accuracy: 0.9578
# 17 Validation accuracy: 0.9598
# 18 Validation accuracy: 0.9588
# 19 Validation accuracy: 0.9604


# Caching the Frozen Layers
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # reused
n_hidden2 = 50  # reused
n_hidden3 = 50  # reused
n_hidden4 = 20  # new!
n_outputs = 10  # new!

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1") # reused frozen
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2") # reused frozen & cached
    hidden2_stop = tf.stop_gradient(hidden2)
    hidden3 = tf.layers.dense(hidden2_stop, n_hidden3, activation=tf.nn.relu, name="hidden3") # reused, not frozen
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4") # new!
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs") # new!

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="hidden[123]") # regular expression
restore_saver = tf.train.Saver(reuse_vars) # to restore layers 1-3

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_batches = len(X_train) // batch_size

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")

    h2_cache = sess.run(hidden2, feed_dict={X: X_train})
    h2_cache_valid = sess.run(hidden2, feed_dict={X: X_valid})  # not shown in the book

    for epoch in range(n_epochs):
        shuffled_idx = np.random.permutation(len(X_train))
        hidden2_batches = np.array_split(h2_cache[shuffled_idx], n_batches)
        y_batches = np.array_split(y_train[shuffled_idx], n_batches)
        for hidden2_batch, y_batch in zip(hidden2_batches, y_batches):
            sess.run(training_op, feed_dict={hidden2: hidden2_batch, y: y_batch})

        accuracy_val = accuracy.eval(feed_dict={hidden2: h2_cache_valid,  # not shown
                                                y: y_valid})  # not shown
        print(epoch, "Validation accuracy:", accuracy_val)  # not shown

    save_path = saver.save(sess, "./my_new_model_final.ckpt")

# 0 Validation accuracy: 0.9018
# 1 Validation accuracy: 0.9306
# 2 Validation accuracy: 0.9432
# 3 Validation accuracy: 0.9476
# 4 Validation accuracy: 0.9516
# 5 Validation accuracy: 0.9522
# 6 Validation accuracy: 0.9522
# 7 Validation accuracy: 0.9554
# 8 Validation accuracy: 0.9552
# 9 Validation accuracy: 0.9558
# 10 Validation accuracy: 0.9572
# 11 Validation accuracy: 0.955
# 12 Validation accuracy: 0.9574
# 13 Validation accuracy: 0.9578
# 14 Validation accuracy: 0.9584
# 15 Validation accuracy: 0.9574
# 16 Validation accuracy: 0.9564
# 17 Validation accuracy: 0.9576
# 18 Validation accuracy: 0.9592
# 19 Validation accuracy: 0.9582


#%% Faster Optimizers

# Momentum optimization
# Momentum optimization cares a great deal about what previous gradients were: at
# each iteration, it adds the local gradient to the momentum vector m (multiplied by the
# learning rate η), and it updates the weights by simply subtracting this momentum vector

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

# Nesterov Accelerated Gradient
# NAG will almost always speed up training compared to regular Momentum optimization.

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9, use_nesterov=True)

# AdaGrad
# you should not use it to train deep neural networks
# (it may be efficient for simpler tasks such as Linear Regression, though)

optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

# RMSProp
# Aunque AdaGrad se ralentiza demasiado rápido y nunca converge al óptimo global,
# el algoritmo RMSProp corrige esto al acumular solo los gradientes de las iteraciones más recientes
# (a diferencia de todos los gradientes desde el comienzo del entrenamiento).
# Lo hace al usar decaimiento exponencial en el primer paso.

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                      momentum=0.9, decay=0.9, epsilon=1e-10)

# Adam Optimization
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


#%% Learning Rate Scheduling
# Since AdaGrad, RMSProp, and Adam optimization automatically reduce the learning
# rate during training, it is not necessary to add an extra learning schedule. For other
# optimization algorithms, using exponential decay or performance scheduling can
# considerably speed up convergence.

reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):       # not shown in the book
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               decay_steps, decay_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")

# 0 Validation accuracy: 0.9568
# 1 Validation accuracy: 0.9728
# 2 Validation accuracy: 0.9732
# 3 Validation accuracy: 0.983
# 4 Validation accuracy: 0.9814


#%% Avoiding Overfitting Through Regularization

# ℓ1 and ℓ2 regularization
# Let's implement ℓ1 regularization manually. First, we create the model, as usual
# (with just one hidden layer this time, for simplicity):

reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    logits = tf.layers.dense(hidden1, n_outputs, name="outputs")

# Next, we get a handle on the layer weights, and we compute the total loss,
# which is equal to the sum of the usual cross entropy loss and the ℓ1 loss
# (i.e., the absolute values of the weights):
W1 = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
W2 = tf.get_default_graph().get_tensor_by_name("outputs/kernel:0")

scale = 0.001 # l1 regularization hyperparameter

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
    reg_losses = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2))
    loss = tf.add(base_loss, scale * reg_losses, name="loss")

# The rest is just as usual:

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")

# 0 Validation accuracy: 0.831
# 1 Validation accuracy: 0.871
# 2 Validation accuracy: 0.8838
# 3 Validation accuracy: 0.8934
# 4 Validation accuracy: 0.8966
# 5 Validation accuracy: 0.8988
# 6 Validation accuracy: 0.9016
# 7 Validation accuracy: 0.9044
# 8 Validation accuracy: 0.9058
# 9 Validation accuracy: 0.906
# 10 Validation accuracy: 0.9068
# 11 Validation accuracy: 0.9054
# 12 Validation accuracy: 0.907
# 13 Validation accuracy: 0.9084
# 14 Validation accuracy: 0.9088
# 15 Validation accuracy: 0.9064
# 16 Validation accuracy: 0.9066
# 17 Validation accuracy: 0.9066
# 18 Validation accuracy: 0.9066
# 19 Validation accuracy: 0.9052

# Alternatively, we can pass a regularization function to the tf.layers.dense() function,
# which will use it to create operations that will compute the regularization loss,
# and it adds these operations to the collection of regularization losses.
# The beginning is the same as above:

reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

# Next, we will use Python's partial() function to avoid repeating the same arguments
# over and over again. Note that we set the kernel_regularizer argument:
scale = 0.001

my_dense_layer = partial(
    tf.layers.dense, activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))

with tf.name_scope("dnn"):
    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    hidden2 = my_dense_layer(hidden1, n_hidden2, name="hidden2")
    logits = my_dense_layer(hidden2, n_outputs, activation=None,
                            name="outputs")

# Next we must add the regularization losses to the base loss:

with tf.name_scope("loss"):                                     # not shown in the book
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  # not shown
        labels=y, logits=logits)                                # not shown
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")   # not shown
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")

# And the rest is the same as usual:

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")

# 0 Validation accuracy: 0.8274
# 1 Validation accuracy: 0.8766
# 2 Validation accuracy: 0.8952
# 3 Validation accuracy: 0.9016
# 4 Validation accuracy: 0.908
# 5 Validation accuracy: 0.9096
# 6 Validation accuracy: 0.9124
# 7 Validation accuracy: 0.9154
# 8 Validation accuracy: 0.9178
# 9 Validation accuracy: 0.919
# 10 Validation accuracy: 0.92
# 11 Validation accuracy: 0.9224
# 12 Validation accuracy: 0.9212
# 13 Validation accuracy: 0.9228
# 14 Validation accuracy: 0.9222
# 15 Validation accuracy: 0.9216
# 16 Validation accuracy: 0.9218
# 17 Validation accuracy: 0.9228
# 18 Validation accuracy: 0.9216
# 19 Validation accuracy: 0.9214


# Dropout
# It is a fairly simple algorithm: at every training step, every neuron (including the
# input neurons but excluding the output neurons) has a probability p of being temporarily
# “dropped out,” meaning it will be entirely ignored during this training step,
# but it may be active during the next step. The hyperparameter p is
# called the dropout rate, and it is typically set to 50%.

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

training = tf.placeholder_with_default(False, shape=(), name='training')

dropout_rate = 0.5  # == 1 - keep_prob
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,
                              name="hidden1")
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu,
                              name="hidden2")
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")

# 0 Validation accuracy: 0.9254
# 1 Validation accuracy: 0.9452
# 2 Validation accuracy: 0.9492
# 3 Validation accuracy: 0.9552
# 4 Validation accuracy: 0.9606
# 5 Validation accuracy: 0.958
# 6 Validation accuracy: 0.9608
# 7 Validation accuracy: 0.966
# 8 Validation accuracy: 0.9694
# 9 Validation accuracy: 0.9708
# 10 Validation accuracy: 0.97
# 11 Validation accuracy: 0.9676
# 12 Validation accuracy: 0.971
# 13 Validation accuracy: 0.9706
# 14 Validation accuracy: 0.9718
# 15 Validation accuracy: 0.9716
# 16 Validation accuracy: 0.972
# 17 Validation accuracy: 0.9718
# 18 Validation accuracy: 0.9728
# 19 Validation accuracy: 0.9736


# Max-Norm Regularization
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

learning_rate = 0.01
momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Next, let's get a handle on the first hidden layer's weight and create an operation
# that will compute the clipped weights using the clip_by_norm() function.
# Then we create an assignment operation to assign the clipped weights to the weights variable:

threshold = 1.0
weights = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
clipped_weights = tf.clip_by_norm(weights, clip_norm=threshold, axes=1)
clip_weights = tf.assign(weights, clipped_weights)

# We can do this as well for the second hidden layer:

weights2 = tf.get_default_graph().get_tensor_by_name("hidden2/kernel:0")
clipped_weights2 = tf.clip_by_norm(weights2, clip_norm=threshold, axes=1)
clip_weights2 = tf.assign(weights2, clipped_weights2)

# Let's add an initializer and a saver:

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# And now we can train the model. It's pretty much as usual,
# except that right after running the training_op,
# we run the clip_weights and clip_weights2 operations:

n_epochs = 20
batch_size = 50

with tf.Session() as sess:                                              # not shown in the book
    init.run()                                                          # not shown
    for epoch in range(n_epochs):                                       # not shown
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size): # not shown
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            clip_weights.eval()
            clip_weights2.eval()                                        # not shown
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})   # not shown
        print(epoch, "Validation accuracy:", acc_valid)                 # not shown

    save_path = saver.save(sess, "./my_model_final.ckpt")               # not shown

# 0 Validation accuracy: 0.9566
# 1 Validation accuracy: 0.9696
# 2 Validation accuracy: 0.9712
# 3 Validation accuracy: 0.9766
# 4 Validation accuracy: 0.977
# 5 Validation accuracy: 0.9776
# 6 Validation accuracy: 0.9816
# 7 Validation accuracy: 0.9812
# 8 Validation accuracy: 0.9798
# 9 Validation accuracy: 0.9818
# 10 Validation accuracy: 0.981
# 11 Validation accuracy: 0.9836
# 12 Validation accuracy: 0.9822
# 13 Validation accuracy: 0.9842
# 14 Validation accuracy: 0.9838
# 15 Validation accuracy: 0.9838
# 16 Validation accuracy: 0.9826
# 17 Validation accuracy: 0.984
# 18 Validation accuracy: 0.9842
# 19 Validation accuracy: 0.984

# The implementation above is straightforward and it works fine, but it is a bit messy.
# A better approach is to define a max_norm_regularizer() function:

def max_norm_regularizer(threshold, axes=1, name="max_norm", collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None # there is no regularization loss term
    return max_norm

# Then you can call this function to get a max norm regularizer (with the threshold you want).
# When you create a hidden layer, you can pass this regularizer to the kernel_regularizer argument:

reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

learning_rate = 0.01
momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

max_norm_reg = max_norm_regularizer(threshold=1.0)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                              kernel_regularizer=max_norm_reg, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
                              kernel_regularizer=max_norm_reg, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Training is as usual, except you must run the weights clipping operations
# after each training operation:

n_epochs = 20
batch_size = 50

clip_all_weights = tf.get_collection("max_norm")

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            sess.run(clip_all_weights)
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid}) # not shown
        print(epoch, "Validation accuracy:", acc_valid)               # not shown

    save_path = saver.save(sess, "./my_model_final.ckpt")             # not shown

# 0 Validation accuracy: 0.9556
# 1 Validation accuracy: 0.9706
# 2 Validation accuracy: 0.9682
# 3 Validation accuracy: 0.9726
# 4 Validation accuracy: 0.9766
# 5 Validation accuracy: 0.976
# 6 Validation accuracy: 0.981
# 7 Validation accuracy: 0.9798
# 8 Validation accuracy: 0.9838
# 9 Validation accuracy: 0.9824
# 10 Validation accuracy: 0.9814
# 11 Validation accuracy: 0.9832
# 12 Validation accuracy: 0.983
# 13 Validation accuracy: 0.9832
# 14 Validation accuracy: 0.9838
# 15 Validation accuracy: 0.9842
# 16 Validation accuracy: 0.9834
# 17 Validation accuracy: 0.984
# 18 Validation accuracy: 0.9838
# 19 Validation accuracy: 0.9836

