# Train a deep MLP on the MNIST dataset and see if you can get over 98% precision.
# Just like in the last exercise of chapter 9,
# try adding all the bells and whistles
# (i.e., save checkpoints, restore the last checkpoint in case of an interruption,
# add summaries, plot learning curves using TensorBoard, and so on).


# First let's create the deep net. It's exactly the same as earlier,
# with just one addition: we add a tf.summary.scalar() to track the loss
# and the accuracy during training, so we can view nice learning curves using TensorBoard.
import numpy as np
import tensorflow as tf
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Now we need to define the directory to write the TensorBoard logs to:
from datetime import datetime

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

logdir = log_dir("mnist_dnn")

# Now we can create the FileWriter that we will use to write the TensorBoard logs:
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Hey! Why don't we implement early stopping?
# For this, we are going to need to use the validation set.
m, n = X_train.shape

n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_deep_mnist_model"

best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


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
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run([accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)
        if epoch % 5 == 0:
            print("Epoch:", epoch,
                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                  "\tLoss: {:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break

# Epoch: 0 	Validation accuracy: 90.240% 	Loss: 0.35380
# Epoch: 5 	Validation accuracy: 95.120% 	Loss: 0.17919
# Epoch: 10 	Validation accuracy: 96.520% 	Loss: 0.12785
# Epoch: 15 	Validation accuracy: 97.180% 	Loss: 0.10326
# Epoch: 20 	Validation accuracy: 97.500% 	Loss: 0.09163
# Epoch: 25 	Validation accuracy: 97.620% 	Loss: 0.08210
# Epoch: 30 	Validation accuracy: 97.740% 	Loss: 0.07883
# Epoch: 35 	Validation accuracy: 97.780% 	Loss: 0.07427
# Epoch: 40 	Validation accuracy: 97.820% 	Loss: 0.07159
# Epoch: 45 	Validation accuracy: 98.080% 	Loss: 0.06740
# Epoch: 50 	Validation accuracy: 98.020% 	Loss: 0.06734
# Epoch: 55 	Validation accuracy: 97.980% 	Loss: 0.06678
# Epoch: 60 	Validation accuracy: 98.020% 	Loss: 0.06731
# Epoch: 65 	Validation accuracy: 98.180% 	Loss: 0.06668
# Epoch: 70 	Validation accuracy: 98.160% 	Loss: 0.06607
# Epoch: 75 	Validation accuracy: 98.120% 	Loss: 0.06644
# Epoch: 80 	Validation accuracy: 98.140% 	Loss: 0.06665
# Epoch: 85 	Validation accuracy: 98.260% 	Loss: 0.06608
# Epoch: 90 	Validation accuracy: 98.200% 	Loss: 0.06731
# Epoch: 95 	Validation accuracy: 98.160% 	Loss: 0.06886
# Epoch: 100 	Validation accuracy: 98.240% 	Loss: 0.06871
# Epoch: 105 	Validation accuracy: 98.240% 	Loss: 0.07066
# Epoch: 110 	Validation accuracy: 98.180% 	Loss: 0.07051
# Epoch: 115 	Validation accuracy: 98.280% 	Loss: 0.07052
# Epoch: 120 	Validation accuracy: 98.240% 	Loss: 0.07304
# Early stopping

os.remove(checkpoint_epoch_path)

with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})

accuracy_val
# Out[11]: 0.9792

# tensorboard --logdir=D:\tf_logs