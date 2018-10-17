# Common imports
import numpy as np
import os
import tensorflow as tf

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#%% Basic RNNs
# Manual RNN
reset_graph()

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

init = tf.global_variables_initializer()

# To run the model, we need to feed it the inputs at both time steps
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

# This mini-batch contains four instances, each with an input sequence composed of
# exactly two inputs

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

print(Y0_val)  # output at t = 0
# [[-0.0664006   0.9625767   0.68105793  0.7091854  -0.898216  ]   # instance 0
#  [ 0.9977755  -0.719789   -0.9965761   0.9673924  -0.9998972 ]   # instance 1
#  [ 0.99999774 -0.99898803 -0.9999989   0.9967762  -0.9999999 ]   # instance 2
#  [ 1.         -1.         -1.         -0.99818915  0.9995087 ]]  # instance 3

print(Y1_val)  # output at t = 1
# [[ 1.         -1.         -1.          0.4020025  -0.9999998 ]   # instance 0
#  [-0.12210421  0.6280527   0.9671843  -0.9937122  -0.25839362]   # instance 1
#  [ 0.9999983  -0.9999994  -0.9999975  -0.8594331  -0.9999881 ]   # instance 2
#  [ 0.99928284 -0.99999815 -0.9999058   0.9857963  -0.92205757]]  # instance 3


# Static Unrolling Through Time
# The following code creates the exact same model as the previous one
n_inputs = 3
n_neurons = 5

reset_graph()

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1],
                                                dtype=tf.float32)
Y0, Y1 = output_seqs
# The static_rnn() function returns two objects
# The first is a Python list containing the output tensors for each time step.
# The second is a tensor containing the final states of the network

init = tf.global_variables_initializer()

X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

Y0_val
# Out[6]:
# array([[ 0.30741334, -0.32884315, -0.6542847 , -0.9385059 ,  0.52089024],
#        [ 0.99122757, -0.9542542 , -0.7518079 , -0.9995208 ,  0.9820235 ],
#        [ 0.9999268 , -0.99783254, -0.8247353 , -0.9999963 ,  0.99947774],
#        [ 0.996771  , -0.68750614,  0.8419969 ,  0.9303911 ,  0.8120684 ]],
#       dtype=float32)

Y1_val
# Out[7]:
# array([[ 0.99998885, -0.9997605 , -0.06679298, -0.9999804 ,  0.99982214],
#        [-0.6524944 , -0.51520866, -0.37968954, -0.59225935, -0.08968385],
#        [ 0.998624  , -0.997152  , -0.03308626, -0.9991565 ,  0.9932902 ],
#        [ 0.99681675, -0.9598194 ,  0.39660636, -0.8307605 ,  0.7967197 ]],
#       dtype=float32)


# Let’s simplify this
# Packing sequences
# The following code builds the same RNN again, but this time it takes a single input
# placeholder of shape [None, n_steps, n_inputs] where the first dimension is the
# mini-batch size
n_steps = 2
n_inputs = 3
n_neurons = 5

reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
# X_seqs is a Python list of n_steps tensors of shape [None, n_inputs], where once again the
# first dimension is the mini-batch size.

X_seqs
# Out[9]:
# [<tf.Tensor 'unstack:0' shape=(?, 3) dtype=float32>,
#  <tf.Tensor 'unstack:1' shape=(?, 3) dtype=float32>]

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs,
                                                dtype=tf.float32)
outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

init = tf.global_variables_initializer()

# Now we can run the network by feeding it a single tensor
# that contains all the minibatch sequences

X_batch = np.array([
        # t = 0      t = 1
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})

# we get a single outputs_val tensor for all instances, all time steps, and all neurons

print(outputs_val)
# [[[-0.45652324 -0.68064123  0.40938237  0.63104504 -0.45732826]
#   [-0.94288003 -0.9998869   0.94055814  0.9999985  -0.9999997 ]]
#  [[-0.8001535  -0.9921827   0.7817797   0.9971031  -0.9964609 ]
#   [-0.637116    0.11300932  0.5798437   0.43105593 -0.63716984]]
#  [[-0.93605185 -0.9998379   0.9308867   0.9999815  -0.99998295]
#   [-0.9165386  -0.9945604   0.89605415  0.99987197 -0.9999751 ]]
#  [[ 0.9927369  -0.9981933  -0.55543643  0.9989031  -0.9953323 ]
#   [-0.02746334 -0.73191994  0.7827872   0.9525682  -0.97817713]]]

print(np.transpose(outputs_val, axes=[1, 0, 2])[1])
# [[-0.94288003 -0.9998869   0.94055814  0.9999985  -0.9999997 ]
#  [-0.637116    0.11300932  0.5798437   0.43105593 -0.63716984]
#  [-0.9165386  -0.9945604   0.89605415  0.99987197 -0.9999751 ]
#  [-0.02746334 -0.73191994  0.7827872   0.9525682  -0.97817713]]


# Dynamic Unrolling Through Time
# The following code creates the same RNN as earlier
# using the dynamic_rnn() function. It’s so much nicer!

n_steps = 2
n_inputs = 3
n_neurons = 5

reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

init = tf.global_variables_initializer()

X_batch = np.array([
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})

print(outputs_val)
# [[[-0.85115266  0.87358344  0.5802911   0.8954789  -0.0557505 ]
#   [-0.9999959   0.9999958   0.9981815   1.          0.37679598]]
#  [[-0.9983293   0.9992038   0.98071456  0.999985    0.2519265 ]
#   [-0.70818055 -0.07723375 -0.8522789   0.5845348  -0.7878095 ]]
#  [[-0.9999827   0.99999535  0.9992863   1.          0.5159071 ]
#   [-0.9993956   0.9984095   0.83422637  0.9999999  -0.47325212]]
#  [[ 0.87888587  0.07356028  0.97216916  0.9998546  -0.7351168 ]
#   [-0.91345143  0.36009577  0.7624866   0.99817705  0.80142   ]]]


# Handling Variable Length Input Sequences
# Setting the sequence lengths
n_steps = 2
n_inputs = 3
n_neurons = 5

reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

seq_length = tf.placeholder(tf.int32, [None])
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,
                                    sequence_length=seq_length)

init = tf.global_variables_initializer()

X_batch = np.array([
        # step 0     step 1
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2 (padded with zero vectors)
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])
seq_length_batch = np.array([2, 1, 2, 2])

with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run(
        [outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})

print(outputs_val)
# [[[-0.9123188   0.16516446  0.5548655  -0.39159346  0.20846416]
#   [-1.          0.9567259   0.99831694  0.99970174  0.9651857 ]]
#  [[-0.9998612   0.6702289   0.9723653   0.6631046   0.74457586]
#   [ 0.          0.          0.          0.          0.        ]]  # zero vector
#  [[-0.99999976  0.8967997   0.9986295   0.9647514   0.9366201 ]
#   [-0.9999526   0.9681953   0.9600286   0.9870625   0.8545923 ]]
#  [[-0.96435434  0.99501586 -0.36150697  0.9983378   0.999497  ]
#   [-0.9613586   0.9568762   0.71322876  0.97729224 -0.09582978]]]

print(states_val)
# [[-1.          0.9567259   0.99831694  0.99970174  0.9651857 ]
#  [-0.9998612   0.6702289   0.9723653   0.6631046   0.74457586]
#  [-0.9999526   0.9681953   0.9600286   0.9870625   0.8545923 ]
#  [-0.9613586   0.9568762   0.71322876  0.97729224 -0.09582978]]


#%% Training RNNs
# To train an RNN, the trick is to unroll it through time (like we just did) and then
# simply use regular backpropagation. This strategy is called backpropagation
# through time (BPTT).
reset_graph()

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

n_epochs = 100
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

# 0 Train accuracy: 0.93333334 Test accuracy: 0.9311
# 1 Train accuracy: 0.96666664 Test accuracy: 0.9522
# 2 Train accuracy: 0.97333336 Test accuracy: 0.9584
# 3 Train accuracy: 0.96 Test accuracy: 0.9613
# 4 Train accuracy: 0.96666664 Test accuracy: 0.9659
# 5 Train accuracy: 0.96666664 Test accuracy: 0.9694
# 6 Train accuracy: 0.97333336 Test accuracy: 0.9692
# 7 Train accuracy: 0.97333336 Test accuracy: 0.9741
# 8 Train accuracy: 0.9533333 Test accuracy: 0.972
# 9 Train accuracy: 0.98 Test accuracy: 0.973
# 10 Train accuracy: 0.98 Test accuracy: 0.972
# 11 Train accuracy: 0.97333336 Test accuracy: 0.9675
# 12 Train accuracy: 0.98 Test accuracy: 0.9707
# 13 Train accuracy: 0.97333336 Test accuracy: 0.9732
# 14 Train accuracy: 0.97333336 Test accuracy: 0.9734
# 15 Train accuracy: 0.9866667 Test accuracy: 0.9729
# 16 Train accuracy: 1.0 Test accuracy: 0.9717
# 17 Train accuracy: 0.9866667 Test accuracy: 0.9732
# 18 Train accuracy: 0.98 Test accuracy: 0.9746
# 19 Train accuracy: 0.9866667 Test accuracy: 0.9751
# 20 Train accuracy: 0.98 Test accuracy: 0.978
# 21 Train accuracy: 0.98 Test accuracy: 0.9765
# 22 Train accuracy: 0.97333336 Test accuracy: 0.9798
# 23 Train accuracy: 0.98 Test accuracy: 0.9693
# 24 Train accuracy: 0.9866667 Test accuracy: 0.9761
# 25 Train accuracy: 0.99333334 Test accuracy: 0.9722
# 26 Train accuracy: 0.96666664 Test accuracy: 0.9767
# 27 Train accuracy: 1.0 Test accuracy: 0.9767
# 28 Train accuracy: 0.98 Test accuracy: 0.9771
# 29 Train accuracy: 1.0 Test accuracy: 0.9769
# 30 Train accuracy: 0.99333334 Test accuracy: 0.9778
# 31 Train accuracy: 0.9866667 Test accuracy: 0.9783
# 32 Train accuracy: 0.98 Test accuracy: 0.9715
# 33 Train accuracy: 0.99333334 Test accuracy: 0.9773
# 34 Train accuracy: 0.9866667 Test accuracy: 0.9785
# 35 Train accuracy: 1.0 Test accuracy: 0.9777
# 36 Train accuracy: 1.0 Test accuracy: 0.9792
# 37 Train accuracy: 0.99333334 Test accuracy: 0.9774
# 38 Train accuracy: 0.9866667 Test accuracy: 0.9785
# 39 Train accuracy: 0.9866667 Test accuracy: 0.9769
# 40 Train accuracy: 0.98 Test accuracy: 0.977
# 41 Train accuracy: 0.99333334 Test accuracy: 0.9785
# 42 Train accuracy: 0.99333334 Test accuracy: 0.9765
# 43 Train accuracy: 0.99333334 Test accuracy: 0.9727
# 44 Train accuracy: 0.97333336 Test accuracy: 0.9741
# 45 Train accuracy: 0.98 Test accuracy: 0.9793
# 46 Train accuracy: 0.99333334 Test accuracy: 0.9803
# 47 Train accuracy: 0.99333334 Test accuracy: 0.9749
# 48 Train accuracy: 0.99333334 Test accuracy: 0.9793
# 49 Train accuracy: 1.0 Test accuracy: 0.9778
# 50 Train accuracy: 0.99333334 Test accuracy: 0.9769
# 51 Train accuracy: 0.9866667 Test accuracy: 0.9724
# 52 Train accuracy: 0.9866667 Test accuracy: 0.9714
# 53 Train accuracy: 0.99333334 Test accuracy: 0.9796
# 54 Train accuracy: 0.9866667 Test accuracy: 0.9759
# 55 Train accuracy: 1.0 Test accuracy: 0.9788
# 56 Train accuracy: 0.99333334 Test accuracy: 0.9787
# 57 Train accuracy: 0.99333334 Test accuracy: 0.9771
# 58 Train accuracy: 0.97333336 Test accuracy: 0.9685
# 59 Train accuracy: 0.99333334 Test accuracy: 0.9804
# 60 Train accuracy: 1.0 Test accuracy: 0.98
# 61 Train accuracy: 1.0 Test accuracy: 0.9794
# 62 Train accuracy: 0.99333334 Test accuracy: 0.9791
# 63 Train accuracy: 0.9866667 Test accuracy: 0.9796
# 64 Train accuracy: 0.99333334 Test accuracy: 0.9777
# 65 Train accuracy: 1.0 Test accuracy: 0.9803
# 66 Train accuracy: 1.0 Test accuracy: 0.9813
# 67 Train accuracy: 0.9866667 Test accuracy: 0.9787
# 68 Train accuracy: 1.0 Test accuracy: 0.9794
# 69 Train accuracy: 0.99333334 Test accuracy: 0.9798
# 70 Train accuracy: 0.9866667 Test accuracy: 0.9811
# 71 Train accuracy: 1.0 Test accuracy: 0.975
# 72 Train accuracy: 1.0 Test accuracy: 0.9763
# 73 Train accuracy: 1.0 Test accuracy: 0.98
# 74 Train accuracy: 0.99333334 Test accuracy: 0.9762
# 75 Train accuracy: 0.99333334 Test accuracy: 0.9768
# 76 Train accuracy: 0.9866667 Test accuracy: 0.9755
# 77 Train accuracy: 1.0 Test accuracy: 0.9786
# 78 Train accuracy: 0.99333334 Test accuracy: 0.9771
# 79 Train accuracy: 0.99333334 Test accuracy: 0.9746
# 80 Train accuracy: 0.98 Test accuracy: 0.9778
# 81 Train accuracy: 1.0 Test accuracy: 0.9756
# 82 Train accuracy: 0.99333334 Test accuracy: 0.9777
# 83 Train accuracy: 0.99333334 Test accuracy: 0.9799
# 84 Train accuracy: 1.0 Test accuracy: 0.9789
# 85 Train accuracy: 1.0 Test accuracy: 0.9767
# 86 Train accuracy: 1.0 Test accuracy: 0.977
# 87 Train accuracy: 0.9866667 Test accuracy: 0.9756
# 88 Train accuracy: 0.99333334 Test accuracy: 0.9781
# 89 Train accuracy: 0.9866667 Test accuracy: 0.9786
# 90 Train accuracy: 1.0 Test accuracy: 0.9769
# 91 Train accuracy: 1.0 Test accuracy: 0.9782
# 92 Train accuracy: 0.99333334 Test accuracy: 0.9775
# 93 Train accuracy: 1.0 Test accuracy: 0.975
# 94 Train accuracy: 1.0 Test accuracy: 0.9787
# 95 Train accuracy: 0.97333336 Test accuracy: 0.9756
# 96 Train accuracy: 0.99333334 Test accuracy: 0.9806
# 97 Train accuracy: 1.0 Test accuracy: 0.9789
# 98 Train accuracy: 0.98 Test accuracy: 0.967
# 99 Train accuracy: 1.0 Test accuracy: 0.9784

