# 10. Pretraining on an auxiliary task

# In this exercise you will build a DNN that compares two MNIST digit images and predicts
# whether they represent the same digit or not.
# Then you will reuse the lower layers of this network to train an MNIST classifier
# using very little training data.

# 10.1 Exercise: Start by building two DNNs (let's call them DNN A and B),
# both similar to the one you built earlier but without the output layer:
# each DNN should have five hidden layers of 100 neurons each, He initialization, and ELU activation.
# Next, add one more hidden layer with 10 units on top of both DNNs.
# You should use TensorFlow's concat() function with axis=1 to concatenate the outputs of both DNNs
# along the horizontal axis, then feed the result to the hidden layer.
# Finally, add an output layer with a single neuron using the logistic activation function.

# You could have two input placeholders, X1 and X2,
# one for the images that should be fed to the first DNN,
# and the other for the images that should be fed to the second DNN.
# It would work fine. However, another option is to have a single input placeholder
# to hold both sets of images (each row will hold a pair of images), and use tf.unstack()
# to split this tensor into two separate tensors, like this:

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

n_inputs = 28 * 28 # MNIST

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, 2, n_inputs), name="X")
X1, X2 = tf.unstack(X, axis=1)

# We also need the labels placeholder. Each label will be 0 if the images represent different digits,
# or 1 if they represent the same digit:

y = tf.placeholder(tf.int32, shape=[None, 1])

# Now let's feed these inputs through two separate DNNs:

he_init = tf.variance_scaling_initializer()

def dnn(inputs, n_hidden_layers=5, n_neurons=100, name=None,
        activation=tf.nn.elu, initializer=he_init):
    with tf.variable_scope(name, "dnn"):
        for layer in range(n_hidden_layers):
            inputs = tf.layers.dense(inputs, n_neurons, activation=activation,
                                     kernel_initializer=initializer,
                                     name="hidden%d" % (layer + 1))
        return inputs

dnn1 = dnn(X1, name="DNN_A")
dnn2 = dnn(X2, name="DNN_B")

# And let's concatenate their outputs:

dnn_outputs = tf.concat([dnn1, dnn2], axis=1)

# Each DNN outputs 100 activations (per instance), so the shape is [None, 100]:

dnn1.shape
# Out[4]: TensorShape([Dimension(None), Dimension(100)])

dnn2.shape
# Out[5]: TensorShape([Dimension(None), Dimension(100)])

# And of course the concatenated outputs have a shape of [None, 200]:

dnn_outputs.shape
# Out[6]: TensorShape([Dimension(None), Dimension(200)])

# Now lets add an extra hidden layer with just 10 neurons, and the output layer, with a single neuron:

hidden = tf.layers.dense(dnn_outputs, units=10, activation=tf.nn.elu, kernel_initializer=he_init)
logits = tf.layers.dense(hidden, units=1, kernel_initializer=he_init)
y_proba = tf.nn.sigmoid(logits)

# The whole network predicts 1 if y_proba >= 0.5
# (i.e. the network predicts that the images represent the same digit), or 0 otherwise.
# We compute instead logits >= 0, which is equivalent but faster to compute:

y_pred = tf.cast(tf.greater_equal(logits, 0), tf.int32)

# Now let's add the cost function:

y_as_float = tf.cast(y, tf.float32)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_as_float, logits=logits)
loss = tf.reduce_mean(xentropy)

# And we can now create the training operation using an optimizer:

learning_rate = 0.01
momentum = 0.95

optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
training_op = optimizer.minimize(loss)

# We will want to measure our classifier's accuracy.

y_pred_correct = tf.equal(y_pred, y)
accuracy = tf.reduce_mean(tf.cast(y_pred_correct, tf.float32))

# And the usual init and saver:

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# 10.2 Exercise: split the MNIST training set in two sets: split #1 should containing 55,000 images,
# and split #2 should contain contain 5,000 images.
# Create a function that generates a training batch where each instance is a pair of MNIST images
# picked from split #1. Half of the training instances should be pairs of images that belong
# to the same class, while the other half should be images from different classes.
# For each pair, the training label should be 0 if the images are from the same class,
# or 1 if they are from different classes.

# The MNIST dataset returned by TensorFlow's input_data() function is already split into 3 parts:
# a training set (55,000 instances), a validation set (5,000 instances)
# and a test set (10,000 instances).
# Let's use the first set to generate the training set composed image pairs,
# and we will use the second set for the second phase of the exercise
# (to train a regular MNIST classifier). We will use the third set as the test set for both phases.

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

X_train1 = X_train
y_train1 = y_train

X_train2 = X_valid
y_train2 = y_valid

X_test = X_test
y_test = y_test

# Let's write a function that generates pairs of images: 50% representing the same digit,
# and 50% representing different digits. There are many ways to implement this.
# In this implementation, we first decide how many "same" pairs
# (i.e. pairs of images representing the same digit) we will generate,
# and how many "different" pairs (i.e. pairs of images representing different digits).
# We could just use batch_size // 2 but we want to handle the case where it is odd
# (granted, that might be overkill!).
# Then we generate random pairs and we pick the right number of "same" pairs,
# then we generate the right number of "different" pairs. Finally we shuffle the batch and return it:

def generate_batch(images, labels, batch_size):
    size1 = batch_size // 2
    size2 = batch_size - size1
    if size1 != size2 and np.random.rand() > 0.5:
        size1, size2 = size2, size1
    X = []
    y = []
    while len(X) < size1:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(images), 2)
        if rnd_idx1 != rnd_idx2 and labels[rnd_idx1] == labels[rnd_idx2]:
            X.append(np.array([images[rnd_idx1], images[rnd_idx2]]))
            y.append([1])
    while len(X) < batch_size:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(images), 2)
        if labels[rnd_idx1] != labels[rnd_idx2]:
            X.append(np.array([images[rnd_idx1], images[rnd_idx2]]))
            y.append([0])
    rnd_indices = np.random.permutation(batch_size)
    return np.array(X)[rnd_indices], np.array(y)[rnd_indices]

# Let's test it to generate a small batch of 5 image pairs:

batch_size = 5
X_batch, y_batch = generate_batch(X_train1, y_train1, batch_size)

# Each row in X_batch contains a pair of images:

X_batch.shape, X_batch.dtype
# Out[16]: ((5, 2, 784), dtype('float32'))

# Let's look at these pairs:

plt.figure(figsize=(3, 3 * batch_size))
plt.subplot(121)
plt.imshow(X_batch[:,0].reshape(28 * batch_size, 28), cmap="binary", interpolation="nearest")
plt.axis('off')
plt.subplot(122)
plt.imshow(X_batch[:,1].reshape(28 * batch_size, 28), cmap="binary", interpolation="nearest")
plt.axis('off')
plt.show()

# And let's look at the labels (0 means "different", 1 means "same"):

y_batch
# Out[19]:
# array([[1],
#        [0],
#        [0],
#        [1],
#        [0]])

# Perfect!


# 10.3 Exercise: train the DNN on this training set. For each image pair,
# you can simultaneously feed the first image to DNN A and the second image to DNN B.
# The whole network will gradually learn to tell whether two images belong to the same class or not.

# Let's generate a test set composed of many pairs of images pulled from the MNIST test set:

X_test1, y_test1 = generate_batch(X_test, y_test, batch_size=len(X_test))

# And now, let's train the model. There's really nothing special about this step,
# except for the fact that we need a fairly large batch_size,
# otherwise the model fails to learn anything and ends up with an accuracy of 50%:

n_epochs = 100
batch_size = 500

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(X_train1) // batch_size):
            X_batch, y_batch = generate_batch(X_train1, y_train1, batch_size)
            loss_val, _ = sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch})
        print(epoch, "Train loss:", loss_val)
        if epoch % 5 == 0:
            acc_test = accuracy.eval(feed_dict={X: X_test1, y: y_test1})
            print(epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_digit_comparison_model.ckpt")

# 0 Train loss: 0.6923601
# 0 Test accuracy: 0.5031
# 1 Train loss: 0.6937516
# 2 Train loss: 0.68896145
# 3 Train loss: 0.6279489
# 4 Train loss: 0.52220035
# 5 Train loss: 0.53943956
# 5 Test accuracy: 0.7317
# 6 Train loss: 0.540305
# 7 Train loss: 0.4539397
# 8 Train loss: 0.4506487
# 9 Train loss: 0.4682584
# 10 Train loss: 0.35033742
# 10 Test accuracy: 0.8226
# 11 Train loss: 0.41334057
# 12 Train loss: 0.36943898
# 13 Train loss: 0.37615353
# 14 Train loss: 0.3200155
# 15 Train loss: 0.32761735
# 15 Test accuracy: 0.8599
# 16 Train loss: 0.31543645
# 17 Train loss: 0.34082982
# 18 Train loss: 0.2899377
# 19 Train loss: 0.30727527
# 20 Train loss: 0.3542515
# 20 Test accuracy: 0.8762
# 21 Train loss: 0.23173124
# 22 Train loss: 0.29069823
# 23 Train loss: 0.24736325
# 24 Train loss: 0.26694357
# 25 Train loss: 0.23917958
# 25 Test accuracy: 0.8854
# 26 Train loss: 0.2561812
# 27 Train loss: 0.30616862
# 28 Train loss: 0.26199126
# 29 Train loss: 0.25882074
# 30 Train loss: 0.26606962
# 30 Test accuracy: 0.8996
# 31 Train loss: 0.26256466
# 32 Train loss: 0.23858154
# 33 Train loss: 0.21879897
# 34 Train loss: 0.25663134
# 35 Train loss: 0.22745427
# 35 Test accuracy: 0.9138
# 36 Train loss: 0.1957055
# 37 Train loss: 0.14133619
# 38 Train loss: 0.15163483
# 39 Train loss: 0.18004777
# 40 Train loss: 0.16298926
# 40 Test accuracy: 0.9302
# 41 Train loss: 0.18049371
# 42 Train loss: 0.15887748
# 43 Train loss: 0.15886801
# 44 Train loss: 0.121250965
# 45 Train loss: 0.18510298
# 45 Test accuracy: 0.9405
# 46 Train loss: 0.12268209
# 47 Train loss: 0.14962167
# 48 Train loss: 0.12870213
# 49 Train loss: 0.09850227
# 50 Train loss: 0.0995818
# 50 Test accuracy: 0.9488
# 51 Train loss: 0.111092165
# 52 Train loss: 0.08176405
# 53 Train loss: 0.1084951
# 54 Train loss: 0.108516544
# 55 Train loss: 0.07292977
# 55 Test accuracy: 0.9566
# 56 Train loss: 0.09244452
# 57 Train loss: 0.087041855
# 58 Train loss: 0.13109061
# 59 Train loss: 0.12440975
# 60 Train loss: 0.08611548
# 60 Test accuracy: 0.9614
# 61 Train loss: 0.082663074
# 62 Train loss: 0.09498146
# 63 Train loss: 0.098821595
# 64 Train loss: 0.10263481
# 65 Train loss: 0.04860875
# 65 Test accuracy: 0.9659
# 66 Train loss: 0.07576445
# 67 Train loss: 0.06883099
# 68 Train loss: 0.06142966
# 69 Train loss: 0.07484267
# 70 Train loss: 0.07764
# 70 Test accuracy: 0.9649
# 71 Train loss: 0.07271394
# 72 Train loss: 0.051152565
# 73 Train loss: 0.047033403
# 74 Train loss: 0.08720791
# 75 Train loss: 0.06833303
# 75 Test accuracy: 0.9695
# 76 Train loss: 0.035827704
# 77 Train loss: 0.05396682
# 78 Train loss: 0.058661755
# 79 Train loss: 0.047542125
# 80 Train loss: 0.040017467
# 80 Test accuracy: 0.9695
# 81 Train loss: 0.055592224
# 82 Train loss: 0.02766697
# 83 Train loss: 0.083333485
# 84 Train loss: 0.033757254
# 85 Train loss: 0.047926515
# 85 Test accuracy: 0.97
# 86 Train loss: 0.024719669
# 87 Train loss: 0.05639553
# 88 Train loss: 0.048691135
# 89 Train loss: 0.044244293
# 90 Train loss: 0.031318232
# 90 Test accuracy: 0.9714
# 91 Train loss: 0.06015031
# 92 Train loss: 0.033373483
# 93 Train loss: 0.03225171
# 94 Train loss: 0.024528554
# 95 Train loss: 0.026843328
# 95 Test accuracy: 0.9724
# 96 Train loss: 0.028708179
# 97 Train loss: 0.02547314
# 98 Train loss: 0.028308537
# 99 Train loss: 0.029124811

# That's not too bad, this model knows a thing or two about comparing handwritten digits!

# Let's see if some of that knowledge can be useful for the regular MNIST classification task.

# 10.4 Exercise: now create a new DNN by reusing and freezing the hidden layers of DNN A
# and adding a softmax output layer on top with 10 neurons.
# Train this network on split #2 and see if you can achieve high performance
# despite having only 500 images per class.

# Let's create the model, it is pretty straightforward.
# There are many ways to freeze the lower layers, as explained in the book.
# In this example, we chose to use the tf.stop_gradient() function.
# Note that we need one Saver to restore the pretrained DNN A,
# and another Saver to save the final model:

reset_graph()

n_inputs = 28 * 28  # MNIST
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

dnn_outputs = dnn(X, name="DNN_A")
frozen_outputs = tf.stop_gradient(dnn_outputs)

logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init)
Y_proba = tf.nn.softmax(logits)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")

optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

dnn_A_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DNN_A")
restore_saver = tf.train.Saver(var_list={var.op.name: var for var in dnn_A_vars})
saver = tf.train.Saver()

# Now on to training! We first initialize all variables
# (including the variables in the new output layer), then we restore the pretrained DNN A.
# Next, we just train the model on the small MNIST dataset (containing just 5,000 images):

n_epochs = 100
batch_size = 50

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_digit_comparison_model.ckpt")

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 10 == 0:
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_mnist_model_final.ckpt")

# 0 Test accuracy: 0.9308
# 10 Test accuracy: 0.9588
# 20 Test accuracy: 0.963
# 30 Test accuracy: 0.9634
# 40 Test accuracy: 0.963
# 50 Test accuracy: 0.9634
# 60 Test accuracy: 0.9632
# 70 Test accuracy: 0.963
# 80 Test accuracy: 0.9631
# 90 Test accuracy: 0.9631

# That's not the best MNIST model we have trained so far,
# but recall that we are only using a small training set (just 500 images per digit).
# Let's compare this result with the same DNN trained from scratch, without using transfer learning:

reset_graph()

n_inputs = 28 * 28  # MNIST
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

dnn_outputs = dnn(X, name="DNN_A")

logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init)
Y_proba = tf.nn.softmax(logits)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")

optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

dnn_A_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DNN_A")
restore_saver = tf.train.Saver(var_list={var.op.name: var for var in dnn_A_vars})
saver = tf.train.Saver()

n_epochs = 150
batch_size = 50

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 10 == 0:
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_mnist_model_final.ckpt")

# 0 Test accuracy: 0.8623
# 10 Test accuracy: 0.9184
# 20 Test accuracy: 0.9382
# 30 Test accuracy: 0.9398
# 40 Test accuracy: 0.9402
# 50 Test accuracy: 0.9406
# 60 Test accuracy: 0.9406
# 70 Test accuracy: 0.9404
# 80 Test accuracy: 0.9403
# 90 Test accuracy: 0.9403
# 100 Test accuracy: 0.9403
# 110 Test accuracy: 0.9401
# 120 Test accuracy: 0.9402
# 130 Test accuracy: 0.9402
# 140 Test accuracy: 0.9402


# Only 94% accuracy... Moreover, the model using transfer learning reached over 96% accuracy in less than 10 epochs.

# Bottom line: transfer learning does not always work (as we saw in exercise 9),
# but when it does it can make a big difference. So try it out!
