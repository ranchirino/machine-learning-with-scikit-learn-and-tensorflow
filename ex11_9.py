# 9.1 Exercise: create a new DNN that reuses all the pretrained hidden layers of the previous model,
# freezes them, and replaces the softmax output layer with a new one.

# Let's load the best model's graph and get a handle on all the important operations we will need.
# Note that instead of creating a new softmax output layer, we will just reuse the existing one
# (since it has the same number of outputs as the existing one).
# We will reinitialize its parameters before training.

import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

restore_saver = tf.train.import_meta_graph("./my_best_mnist_model_0_to_4.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")
loss = tf.get_default_graph().get_tensor_by_name("loss:0")
Y_proba = tf.get_default_graph().get_tensor_by_name("Y_proba:0")
logits = Y_proba.op.inputs[0]
accuracy = tf.get_default_graph().get_tensor_by_name("accuracy:0")

# To freeze the lower layers, we will exclude their variables
# from the optimizer's list of trainable variables, keeping only the output layer's trainable variables:

learning_rate = 0.01

output_layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="logits")
optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam2")
training_op = optimizer.minimize(loss, var_list=output_layer_vars)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init = tf.global_variables_initializer()
five_frozen_saver = tf.train.Saver()


# 9.2 Exercise: train this new DNN on digits 5 to 9, using only 100 images per digit,
# and time how long it takes. Despite this small number of examples, can you achieve high precision?

# Let's create the training, validation and test sets.
# We need to subtract 5 from the labels because TensorFlow expects integers from 0 to n_classes-1.

# Let's load the data:
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

X_train2_full = X_train[y_train >= 5]
y_train2_full = y_train[y_train >= 5] - 5
X_valid2_full = X_valid[y_valid >= 5]
y_valid2_full = y_valid[y_valid >= 5] - 5
X_test2 = X_test[y_test >= 5]
y_test2 = y_test[y_test >= 5] - 5

# Also, for the purpose of this exercise,
# we want to keep only 100 instances per class in the training set
# (and let's keep only 30 instances per class in the validation set).
# Let's create a small function to do that:

def sample_n_instances_per_class(X, y, n=100):
    Xs, ys = [], []
    for label in np.unique(y):
        idx = (y == label)
        Xc = X[idx][:n]
        yc = y[idx][:n]
        Xs.append(Xc)
        ys.append(yc)
    return np.concatenate(Xs), np.concatenate(ys)

X_train2, y_train2 = sample_n_instances_per_class(X_train2_full, y_train2_full, n=100)
X_valid2, y_valid2 = sample_n_instances_per_class(X_valid2_full, y_valid2_full, n=30)

# Now let's train the model. This is the same training code as earlier, using early stopping,
# except for the initialization: we first initialize all the variables,
# then we restore the best model trained earlier (on digits 0 to 4),
# and finally we reinitialize the output layer variables.

import time

n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_best_mnist_model_0_to_4")
    for var in output_layer_vars:
        var.initializer.run()

    t0 = time.time()

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid2, y: y_valid2})
        if loss_val < best_loss:
            save_path = five_frozen_saver.save(sess, "./my_mnist_model_5_to_9_five_frozen")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))

    t1 = time.time()
    print("Total training time: {:.1f}s".format(t1 - t0))

with tf.Session() as sess:
    five_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_five_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test2, y: y_test2})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

# 0	Validation loss: 1.003397	Best loss: 1.003397	Accuracy: 65.33%
# 1	Validation loss: 1.022734	Best loss: 1.003397	Accuracy: 68.00%
# 2	Validation loss: 0.839555	Best loss: 0.839555	Accuracy: 68.00%
# 3	Validation loss: 0.788488	Best loss: 0.788488	Accuracy: 70.67%
# 4	Validation loss: 1.017687	Best loss: 0.788488	Accuracy: 62.00%
# 5	Validation loss: 0.738970	Best loss: 0.738970	Accuracy: 73.33%
# 6	Validation loss: 0.858980	Best loss: 0.738970	Accuracy: 76.67%
# 7	Validation loss: 0.769839	Best loss: 0.738970	Accuracy: 76.67%
# 8	Validation loss: 1.178967	Best loss: 0.738970	Accuracy: 64.67%
# 9	Validation loss: 0.646462	Best loss: 0.646462	Accuracy: 78.67%
# 10	Validation loss: 0.903281	Best loss: 0.646462	Accuracy: 70.67%
# 11	Validation loss: 0.871646	Best loss: 0.646462	Accuracy: 72.00%
# 12	Validation loss: 0.677795	Best loss: 0.646462	Accuracy: 79.33%
# 13	Validation loss: 0.893950	Best loss: 0.646462	Accuracy: 74.67%
# 14	Validation loss: 0.832718	Best loss: 0.646462	Accuracy: 73.33%
# 15	Validation loss: 0.825619	Best loss: 0.646462	Accuracy: 76.67%
# 16	Validation loss: 0.769393	Best loss: 0.646462	Accuracy: 73.33%
# 17	Validation loss: 0.718477	Best loss: 0.646462	Accuracy: 76.67%
# 18	Validation loss: 0.706812	Best loss: 0.646462	Accuracy: 77.33%
# 19	Validation loss: 0.791332	Best loss: 0.646462	Accuracy: 72.67%
# 20	Validation loss: 0.721815	Best loss: 0.646462	Accuracy: 78.00%
# 21	Validation loss: 0.765417	Best loss: 0.646462	Accuracy: 78.00%
# 22	Validation loss: 0.679198	Best loss: 0.646462	Accuracy: 78.67%
# 23	Validation loss: 0.730918	Best loss: 0.646462	Accuracy: 77.33%
# 24	Validation loss: 0.681055	Best loss: 0.646462	Accuracy: 78.67%
# 25	Validation loss: 0.664438	Best loss: 0.646462	Accuracy: 77.33%
# 26	Validation loss: 0.728960	Best loss: 0.646462	Accuracy: 79.33%
# 27	Validation loss: 0.933802	Best loss: 0.646462	Accuracy: 78.00%
# 28	Validation loss: 0.688992	Best loss: 0.646462	Accuracy: 77.33%
# 29	Validation loss: 0.760406	Best loss: 0.646462	Accuracy: 73.33%
# Early stopping!
# Total training time: 5.1s
# INFO:tensorflow:Restoring parameters from ./my_mnist_model_5_to_9_five_frozen
# Final test accuracy: 71.10%

# Well that's not a great accuracy, is it? Of course with such a tiny training set,
# and with only one layer to tweak, we should not expect miracles.

# 9.3 Exercise: try caching the frozen layers, and train the model again: how much faster is it now?

# Let's start by getting a handle on the output of the last frozen layer:

hidden5_out = tf.get_default_graph().get_tensor_by_name("hidden5_out:0")

# Now let's train the model using roughly the same code as earlier.
# The difference is that we compute the output of the top frozen layer at the beginning
# (both for the training set and the validation set), and we cache it.
# This makes training roughly 1.5 to 3 times faster in this example (this may vary greatly,
# depending on your system):

import time

n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_best_mnist_model_0_to_4")
    for var in output_layer_vars:
        var.initializer.run()

    t0 = time.time()

    hidden5_train = hidden5_out.eval(feed_dict={X: X_train2, y: y_train2})
    hidden5_valid = hidden5_out.eval(feed_dict={X: X_valid2, y: y_valid2})

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            h5_batch, y_batch = hidden5_train[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op, feed_dict={hidden5_out: h5_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={hidden5_out: hidden5_valid, y: y_valid2})
        if loss_val < best_loss:
            save_path = five_frozen_saver.save(sess, "./my_mnist_model_5_to_9_five_frozen")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))

    t1 = time.time()
    print("Total training time: {:.1f}s".format(t1 - t0))

with tf.Session() as sess:
    five_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_five_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test2, y: y_test2})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

# 0	Validation loss: 0.981678	Best loss: 0.981678	Accuracy: 64.67%
# 1	Validation loss: 0.878629	Best loss: 0.878629	Accuracy: 71.33%
# 2	Validation loss: 0.815959	Best loss: 0.815959	Accuracy: 70.00%
# 3	Validation loss: 0.835458	Best loss: 0.815959	Accuracy: 78.00%
# 4	Validation loss: 0.849617	Best loss: 0.815959	Accuracy: 70.67%
# 5	Validation loss: 0.738226	Best loss: 0.738226	Accuracy: 76.00%
# 6	Validation loss: 0.852809	Best loss: 0.738226	Accuracy: 78.00%
# 7	Validation loss: 0.737371	Best loss: 0.737371	Accuracy: 70.67%
# 8	Validation loss: 0.673178	Best loss: 0.673178	Accuracy: 78.00%
# 9	Validation loss: 0.762029	Best loss: 0.673178	Accuracy: 74.00%
# 10	Validation loss: 0.686108	Best loss: 0.673178	Accuracy: 77.33%
# 11	Validation loss: 0.686257	Best loss: 0.673178	Accuracy: 75.33%
# 12	Validation loss: 0.799801	Best loss: 0.673178	Accuracy: 64.67%
# 13	Validation loss: 0.771081	Best loss: 0.673178	Accuracy: 74.67%
# 14	Validation loss: 0.849876	Best loss: 0.673178	Accuracy: 74.00%
# 15	Validation loss: 0.693472	Best loss: 0.673178	Accuracy: 77.33%
# 16	Validation loss: 0.664397	Best loss: 0.664397	Accuracy: 77.33%
# 17	Validation loss: 0.909794	Best loss: 0.664397	Accuracy: 74.67%
# 18	Validation loss: 0.773446	Best loss: 0.664397	Accuracy: 73.33%
# 19	Validation loss: 0.744807	Best loss: 0.664397	Accuracy: 76.67%
# 20	Validation loss: 0.626903	Best loss: 0.626903	Accuracy: 81.33%
# 21	Validation loss: 0.882686	Best loss: 0.626903	Accuracy: 71.33%
# 22	Validation loss: 0.724000	Best loss: 0.626903	Accuracy: 78.67%
# 23	Validation loss: 0.823099	Best loss: 0.626903	Accuracy: 74.00%
# 24	Validation loss: 0.716853	Best loss: 0.626903	Accuracy: 76.67%
# 25	Validation loss: 0.801235	Best loss: 0.626903	Accuracy: 73.33%
# 26	Validation loss: 0.924824	Best loss: 0.626903	Accuracy: 78.00%
# 27	Validation loss: 1.013090	Best loss: 0.626903	Accuracy: 67.33%
# 28	Validation loss: 0.790636	Best loss: 0.626903	Accuracy: 77.33%
# 29	Validation loss: 0.854826	Best loss: 0.626903	Accuracy: 74.00%
# 30	Validation loss: 0.809002	Best loss: 0.626903	Accuracy: 74.67%
# 31	Validation loss: 0.880133	Best loss: 0.626903	Accuracy: 72.00%
# 32	Validation loss: 0.709988	Best loss: 0.626903	Accuracy: 77.33%
# 33	Validation loss: 0.761364	Best loss: 0.626903	Accuracy: 72.00%
# 34	Validation loss: 0.718697	Best loss: 0.626903	Accuracy: 76.67%
# 35	Validation loss: 0.774806	Best loss: 0.626903	Accuracy: 74.67%
# 36	Validation loss: 0.895356	Best loss: 0.626903	Accuracy: 71.33%
# 37	Validation loss: 0.754592	Best loss: 0.626903	Accuracy: 76.00%
# 38	Validation loss: 0.756375	Best loss: 0.626903	Accuracy: 72.00%
# 39	Validation loss: 0.669439	Best loss: 0.626903	Accuracy: 77.33%
# 40	Validation loss: 0.660094	Best loss: 0.626903	Accuracy: 78.00%
# Early stopping!
# Total training time: 7.2s
# INFO:tensorflow:Restoring parameters from ./my_mnist_model_5_to_9_five_frozen
# Final test accuracy: 71.12%


# 9.4 Exercise: try again reusing just four hidden layers instead of five.
# Can you achieve a higher precision?

# Let's load the best model again,
# but this time we will create a new softmax output layer on top of the 4th hidden layer:

reset_graph()

n_outputs = 5

restore_saver = tf.train.import_meta_graph("./my_best_mnist_model_0_to_4.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

he_init = tf.variance_scaling_initializer()
hidden4_out = tf.get_default_graph().get_tensor_by_name("hidden4_out:0")
logits = tf.layers.dense(hidden4_out, n_outputs, kernel_initializer=he_init, name="new_logits")
Y_proba = tf.nn.softmax(logits)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# And now let's create the training operation.
# We want to freeze all the layers except for the new output layer:

learning_rate = 0.01

output_layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="new_logits")
optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam2")
training_op = optimizer.minimize(loss, var_list=output_layer_vars)

init = tf.global_variables_initializer()
four_frozen_saver = tf.train.Saver()

# And once again we train the model with the same code as earlier.

n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_best_mnist_model_0_to_4")

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid2, y: y_valid2})
        if loss_val < best_loss:
            save_path = four_frozen_saver.save(sess, "./my_mnist_model_5_to_9_four_frozen")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))

with tf.Session() as sess:
    four_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_four_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test2, y: y_test2})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

# 0	Validation loss: 1.126248	Best loss: 1.126248	Accuracy: 70.00%
# 1	Validation loss: 1.269771	Best loss: 1.126248	Accuracy: 72.67%
# 2	Validation loss: 0.890106	Best loss: 0.890106	Accuracy: 68.00%
# 3	Validation loss: 0.920281	Best loss: 0.890106	Accuracy: 76.00%
# 4	Validation loss: 1.222460	Best loss: 0.890106	Accuracy: 64.00%
# 5	Validation loss: 0.779300	Best loss: 0.779300	Accuracy: 74.67%
# 6	Validation loss: 0.842474	Best loss: 0.779300	Accuracy: 75.33%
# 7	Validation loss: 0.844362	Best loss: 0.779300	Accuracy: 77.33%
# 8	Validation loss: 1.190571	Best loss: 0.779300	Accuracy: 72.67%
# 9	Validation loss: 0.760679	Best loss: 0.760679	Accuracy: 80.00%
# 10	Validation loss: 0.878683	Best loss: 0.760679	Accuracy: 75.33%
# 11	Validation loss: 0.879104	Best loss: 0.760679	Accuracy: 70.67%
# 12	Validation loss: 0.864376	Best loss: 0.760679	Accuracy: 77.33%
# 13	Validation loss: 0.975808	Best loss: 0.760679	Accuracy: 74.00%
# 14	Validation loss: 0.889145	Best loss: 0.760679	Accuracy: 74.67%
# 15	Validation loss: 0.796109	Best loss: 0.760679	Accuracy: 76.67%
# 16	Validation loss: 1.152294	Best loss: 0.760679	Accuracy: 70.00%
# 17	Validation loss: 0.863581	Best loss: 0.760679	Accuracy: 76.00%
# 18	Validation loss: 0.886885	Best loss: 0.760679	Accuracy: 76.67%
# 19	Validation loss: 0.862228	Best loss: 0.760679	Accuracy: 74.67%
# 20	Validation loss: 0.759977	Best loss: 0.759977	Accuracy: 78.67%
# 21	Validation loss: 1.007297	Best loss: 0.759977	Accuracy: 76.67%
# 22	Validation loss: 0.814739	Best loss: 0.759977	Accuracy: 73.33%
# 23	Validation loss: 0.807654	Best loss: 0.759977	Accuracy: 78.67%
# 24	Validation loss: 0.769105	Best loss: 0.759977	Accuracy: 74.00%
# 25	Validation loss: 0.745075	Best loss: 0.745075	Accuracy: 77.33%
# 26	Validation loss: 0.993029	Best loss: 0.745075	Accuracy: 75.33%
# 27	Validation loss: 1.531440	Best loss: 0.745075	Accuracy: 78.67%
# 28	Validation loss: 1.078498	Best loss: 0.745075	Accuracy: 78.67%
# 29	Validation loss: 0.943216	Best loss: 0.745075	Accuracy: 74.00%
# 30	Validation loss: 0.810529	Best loss: 0.745075	Accuracy: 79.33%
# 31	Validation loss: 0.874354	Best loss: 0.745075	Accuracy: 74.00%
# 32	Validation loss: 0.941409	Best loss: 0.745075	Accuracy: 76.67%
# 33	Validation loss: 1.003653	Best loss: 0.745075	Accuracy: 74.00%
# 34	Validation loss: 0.808762	Best loss: 0.745075	Accuracy: 78.00%
# 35	Validation loss: 0.830576	Best loss: 0.745075	Accuracy: 79.33%
# 36	Validation loss: 1.145891	Best loss: 0.745075	Accuracy: 75.33%
# 37	Validation loss: 1.500586	Best loss: 0.745075	Accuracy: 78.67%
# 38	Validation loss: 1.258644	Best loss: 0.745075	Accuracy: 78.67%
# 39	Validation loss: 1.002806	Best loss: 0.745075	Accuracy: 76.00%
# 40	Validation loss: 0.801568	Best loss: 0.745075	Accuracy: 76.67%
# 41	Validation loss: 0.913200	Best loss: 0.745075	Accuracy: 78.67%
# 42	Validation loss: 0.861955	Best loss: 0.745075	Accuracy: 77.33%
# 43	Validation loss: 0.905845	Best loss: 0.745075	Accuracy: 74.67%
# 44	Validation loss: 0.985188	Best loss: 0.745075	Accuracy: 78.00%
# 45	Validation loss: 1.420659	Best loss: 0.745075	Accuracy: 78.00%
# Early stopping!
# INFO:tensorflow:Restoring parameters from ./my_mnist_model_5_to_9_four_frozen
# Final test accuracy: 73.81%

# Still not fantastic, but much better.


# 9.5 Exercise: now unfreeze the top two hidden layers and continue training:
# can you get the model to perform even better?

learning_rate = 0.01

unfrozen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[34]|new_logits")
optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam3")
training_op = optimizer.minimize(loss, var_list=unfrozen_vars)

init = tf.global_variables_initializer()
two_frozen_saver = tf.train.Saver()

n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    four_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_four_frozen")

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid2, y: y_valid2})
        if loss_val < best_loss:
            save_path = two_frozen_saver.save(sess, "./my_mnist_model_5_to_9_two_frozen")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))

with tf.Session() as sess:
    two_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_two_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test2, y: y_test2})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

# 0	Validation loss: 1.172862	Best loss: 1.172862	Accuracy: 80.67%
# 1	Validation loss: 0.978189	Best loss: 0.978189	Accuracy: 81.33%
# 2	Validation loss: 0.571891	Best loss: 0.571891	Accuracy: 82.67%
# 3	Validation loss: 0.819631	Best loss: 0.571891	Accuracy: 82.67%
# 4	Validation loss: 0.777372	Best loss: 0.571891	Accuracy: 84.67%
# 5	Validation loss: 1.017751	Best loss: 0.571891	Accuracy: 79.33%
# 6	Validation loss: 0.615452	Best loss: 0.571891	Accuracy: 85.33%
# 7	Validation loss: 0.861803	Best loss: 0.571891	Accuracy: 82.67%
# 8	Validation loss: 1.160020	Best loss: 0.571891	Accuracy: 76.00%
# 9	Validation loss: 1.073469	Best loss: 0.571891	Accuracy: 82.00%
# 10	Validation loss: 1.105305	Best loss: 0.571891	Accuracy: 82.67%
# 11	Validation loss: 0.933681	Best loss: 0.571891	Accuracy: 85.33%
# 12	Validation loss: 0.935490	Best loss: 0.571891	Accuracy: 81.33%
# 13	Validation loss: 0.816262	Best loss: 0.571891	Accuracy: 84.67%
# 14	Validation loss: 0.790795	Best loss: 0.571891	Accuracy: 86.67%
# 15	Validation loss: 0.869112	Best loss: 0.571891	Accuracy: 84.67%
# 16	Validation loss: 0.825750	Best loss: 0.571891	Accuracy: 85.33%
# 17	Validation loss: 0.849888	Best loss: 0.571891	Accuracy: 86.00%
# 18	Validation loss: 0.919446	Best loss: 0.571891	Accuracy: 86.00%
# 19	Validation loss: 0.897450	Best loss: 0.571891	Accuracy: 85.33%
# 20	Validation loss: 1.007597	Best loss: 0.571891	Accuracy: 85.33%
# 21	Validation loss: 0.945295	Best loss: 0.571891	Accuracy: 86.67%
# 22	Validation loss: 0.895254	Best loss: 0.571891	Accuracy: 84.00%
# Early stopping!
# INFO:tensorflow:Restoring parameters from ./my_mnist_model_5_to_9_two_frozen
# Final test accuracy: 78.65%

# Let's check what accuracy we can get by unfreezing all layers:

learning_rate = 0.01

optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam4")
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
no_frozen_saver = tf.train.Saver()

n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    two_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_two_frozen")

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid2, y: y_valid2})
        if loss_val < best_loss:
            save_path = no_frozen_saver.save(sess, "./my_mnist_model_5_to_9_no_frozen")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))

with tf.Session() as sess:
    no_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_no_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test2, y: y_test2})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

# 0	Validation loss: 0.899511	Best loss: 0.899511	Accuracy: 83.33%
# 1	Validation loss: 2.855118	Best loss: 0.899511	Accuracy: 77.33%
# 2	Validation loss: 7.816149	Best loss: 0.899511	Accuracy: 86.00%
# 3	Validation loss: 13.282640	Best loss: 0.899511	Accuracy: 89.33%
# 4	Validation loss: 1.323414	Best loss: 0.899511	Accuracy: 88.00%
# 5	Validation loss: 3.764161	Best loss: 0.899511	Accuracy: 90.67%
# 6	Validation loss: 8.047982	Best loss: 0.899511	Accuracy: 88.67%
# 7	Validation loss: 3.649852	Best loss: 0.899511	Accuracy: 89.33%
# 8	Validation loss: 4.086418	Best loss: 0.899511	Accuracy: 89.33%
# 9	Validation loss: 8.735205	Best loss: 0.899511	Accuracy: 88.67%
# 10	Validation loss: 1.489866	Best loss: 0.899511	Accuracy: 86.67%
# 11	Validation loss: 1.394844	Best loss: 0.899511	Accuracy: 81.33%
# 12	Validation loss: 1.066774	Best loss: 0.899511	Accuracy: 89.33%
# 13	Validation loss: 1.529656	Best loss: 0.899511	Accuracy: 87.33%
# 14	Validation loss: 0.626035	Best loss: 0.626035	Accuracy: 91.33%
# 15	Validation loss: 2.377292	Best loss: 0.626035	Accuracy: 90.67%
# 16	Validation loss: 1.623098	Best loss: 0.626035	Accuracy: 88.67%
# 17	Validation loss: 1.432189	Best loss: 0.626035	Accuracy: 90.67%
# 18	Validation loss: 2.123802	Best loss: 0.626035	Accuracy: 90.00%
# 19	Validation loss: 2.250925	Best loss: 0.626035	Accuracy: 92.00%
# 20	Validation loss: 2.309876	Best loss: 0.626035	Accuracy: 91.33%
# 21	Validation loss: 2.442607	Best loss: 0.626035	Accuracy: 91.33%
# 22	Validation loss: 2.521206	Best loss: 0.626035	Accuracy: 91.33%
# 23	Validation loss: 2.559074	Best loss: 0.626035	Accuracy: 91.33%
# 24	Validation loss: 2.591397	Best loss: 0.626035	Accuracy: 91.33%
# 25	Validation loss: 2.634803	Best loss: 0.626035	Accuracy: 91.33%
# 26	Validation loss: 2.663750	Best loss: 0.626035	Accuracy: 91.33%
# 27	Validation loss: 2.702619	Best loss: 0.626035	Accuracy: 91.33%
# 28	Validation loss: 2.741448	Best loss: 0.626035	Accuracy: 91.33%
# 29	Validation loss: 2.764320	Best loss: 0.626035	Accuracy: 91.33%
# 30	Validation loss: 2.779500	Best loss: 0.626035	Accuracy: 91.33%
# 31	Validation loss: 2.799955	Best loss: 0.626035	Accuracy: 91.33%
# 32	Validation loss: 2.810244	Best loss: 0.626035	Accuracy: 91.33%
# 33	Validation loss: 2.821529	Best loss: 0.626035	Accuracy: 91.33%
# 34	Validation loss: 2.831756	Best loss: 0.626035	Accuracy: 91.33%
# Early stopping!
# INFO:tensorflow:Restoring parameters from ./my_mnist_model_5_to_9_no_frozen
# Final test accuracy: 87.90%

# Let's compare that to a DNN trained from scratch:
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden_layers=5, n_neurons=100, optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.01, batch_size=20, activation=tf.nn.elu, initializer=he_init,
                 batch_norm_momentum=None, dropout_rate=None, random_state=None):
        """Initialize the DNNClassifier by simply storing all the hyperparameters."""
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None

    def _dnn(self, inputs):
        """Build the hidden layers, with support for batch normalization and dropout."""
        for layer in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs, self.dropout_rate, training=self._training)
            inputs = tf.layers.dense(inputs, self.n_neurons,
                                     kernel_initializer=self.initializer,
                                     name="hidden%d" % (layer + 1))
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum,
                                                       training=self._training)
            inputs = self.activation(inputs, name="hidden%d_out" % (layer + 1))
        return inputs

    def _build_graph(self, n_inputs, n_outputs):
        """Build the same model as earlier"""
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None

        dnn_outputs = self._dnn(X)

        logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        """Get all variable values (used for early stopping, faster than saving to disk)"""
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """Set all variables to the given values (for early stopping, faster than loading from disk)"""
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
        """Fit the model to the training set. If X_valid and y_valid are provided, use early stopping."""
        self.close_session()

        # infer n_inputs and n_outputs from the training set.
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)

        # Translate the labels vector to a vector of sorted class indices, containing
        # integers from 0 to n_outputs - 1.
        # For example, if y is equal to [8, 8, 9, 5, 7, 6, 6, 6], then the sorted class
        # labels (self.classes_) will be equal to [5, 6, 7, 8, 9], and the labels vector
        # will be translated to [3, 3, 4, 0, 2, 1, 1, 1]
        self.class_to_index_ = {label: index
                                for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label]
                      for label in y], dtype=np.int32)

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            # extra ops for batch normalization
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # needed in case of early stopping
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None

        # Now train the model!
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx, len(X) // self.batch_size):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True
                    sess.run(self._training_op, feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops, feed_dict=feed_dict)
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val = sess.run([self._loss, self._accuracy],
                                                 feed_dict={self._X: X_valid,
                                                            self._y: y_valid})
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 1
                    print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_val, best_loss, acc_val * 100))
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping!")
                        break
                else:
                    loss_train, acc_train = sess.run([self._loss, self._accuracy],
                                                     feed_dict={self._X: X_batch,
                                                                self._y: y_batch})
                    print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_train, acc_train * 100))
            # If we used early stopping then rollback to the best model found
            if best_params:
                self._restore_model_params(best_params)
            return self

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices], np.int32)

    def save(self, path):
        self._saver.save(self._session, path)

dnn_clf_5_to_9 = DNNClassifier(n_hidden_layers=4, random_state=42)
dnn_clf_5_to_9.fit(X_train2, y_train2, n_epochs=1000, X_valid=X_valid2, y_valid=y_valid2)

# 0	Validation loss: 0.647537	Best loss: 0.647537	Accuracy: 77.33%
# 1	Validation loss: 0.495277	Best loss: 0.495277	Accuracy: 87.33%
# 2	Validation loss: 0.523580	Best loss: 0.495277	Accuracy: 87.33%
# 3	Validation loss: 0.561892	Best loss: 0.495277	Accuracy: 90.00%
# 4	Validation loss: 0.750305	Best loss: 0.495277	Accuracy: 90.67%
# 5	Validation loss: 0.568648	Best loss: 0.495277	Accuracy: 82.00%
# 6	Validation loss: 0.851868	Best loss: 0.495277	Accuracy: 86.67%
# 7	Validation loss: 1.529336	Best loss: 0.495277	Accuracy: 88.00%
# 8	Validation loss: 2.327442	Best loss: 0.495277	Accuracy: 85.33%
# 9	Validation loss: 3.199109	Best loss: 0.495277	Accuracy: 84.67%
# 10	Validation loss: 0.899644	Best loss: 0.495277	Accuracy: 79.33%
# 11	Validation loss: 1.380145	Best loss: 0.495277	Accuracy: 89.33%
# 12	Validation loss: 2.560563	Best loss: 0.495277	Accuracy: 89.33%
# 13	Validation loss: 3.579709	Best loss: 0.495277	Accuracy: 87.33%
# 14	Validation loss: 3.119990	Best loss: 0.495277	Accuracy: 90.00%
# 15	Validation loss: 5.599546	Best loss: 0.495277	Accuracy: 90.00%
# 16	Validation loss: 1.709901	Best loss: 0.495277	Accuracy: 88.67%
# 17	Validation loss: 1.845016	Best loss: 0.495277	Accuracy: 80.00%
# 18	Validation loss: 1.062110	Best loss: 0.495277	Accuracy: 92.00%
# 19	Validation loss: 1.315856	Best loss: 0.495277	Accuracy: 90.00%
# 20	Validation loss: 1.163990	Best loss: 0.495277	Accuracy: 92.00%
# 21	Validation loss: 1.157072	Best loss: 0.495277	Accuracy: 92.67%
# 22	Validation loss: 0.765040	Best loss: 0.495277	Accuracy: 92.00%
# Early stopping!
# Out[14]:
# DNNClassifier(activation=<function elu at 0x0000023F97BA12F0>,
#        batch_norm_momentum=None, batch_size=20, dropout_rate=None,
#        initializer=<tensorflow.python.ops.init_ops.VarianceScaling object at 0x0000023FA0742278>,
#        learning_rate=0.01, n_hidden_layers=4, n_neurons=100,
#        optimizer_class=<class 'tensorflow.python.training.adam.AdamOptimizer'>,
#        random_state=42)

from sklearn.metrics import accuracy_score

y_pred = dnn_clf_5_to_9.predict(X_test2)
accuracy_score(y_test2, y_pred)
# Out[16]: 0.843859288212302

# Meh. How disappointing! ;) Transfer learning did not help much (if at all) in this task.
# At least we tried... Fortunately, the next exercise will get better results.

