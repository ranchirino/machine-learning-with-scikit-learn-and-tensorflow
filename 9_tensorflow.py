# Creating Your First Graph and Running It in a Session
import numpy as np
import tensorflow as tf

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
# Out[2]: <tf.Tensor 'add_1:0' shape=() dtype=int32>

# To evaluate this graph, you need to open a TensorFlow session and use it
# to initialize the variables and evaluate f
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
# 42
sess.close()

# Having to repeat sess.run() all the time is a bit cumbersome

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

result
# Out[6]: 42

# Inside the with block, the session is set as the default session.
# This makes the code easier to read. Moreover,
# the session is automatically closed at the end of the block.

# Instead of manually running the initializer for every single variable, you can use the
# global_variables_initializer() function
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()

result
# Out[8]: 42

# you may prefer to create an InteractiveSes
# sion. The only difference from a regular Session is that when an InteractiveSes
# sion is created it automatically sets itself as the default session, so you don’t need a
# with block (but you do need to close the session manually when you are done with
# it)
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
# 42
sess.close()

# A TensorFlow program is typically split into two parts: the first part builds a computation
# graph (this is called the construction phase), and the second part runs it (this is
# the execution phase).

#%% Managing Graphs
# Any node you create is automatically added to the default graph:
reset_graph()

x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()
# Out[11]: True

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph
# Out[12]: True

x2.graph is tf.get_default_graph()
# Out[13]: False


#%% Lifecycle of a Node Value
# When you evaluate a node, TensorFlow automatically determines the set of nodes
# that it depends on and it evaluates these nodes first.

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())  # 10
    print(z.eval())  # 15
# 10
# 15

# A variable starts its life when its initializer is run,
# and it ends when the session is closed.
# you must ask TensorFlow to evaluate both y and z in just one graph
# run, as shown in the following code
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)  # 10
    print(z_val)  # 15
# 10
# 15

#%% Linear Regression
# Using the Normal Equation
from sklearn.datasets import fetch_california_housing

reset_graph()

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()

theta_value
# Out[20]:
# array([[-3.7465141e+01],
#        [ 4.3573415e-01],
#        [ 9.3382923e-03],
#        [-1.0662201e-01],
#        [ 6.4410698e-01],
#        [-4.2513184e-06],
#        [-3.7732250e-03],
#        [-4.2664889e-01],
#        [-4.4051403e-01]], dtype=float32)

# The main benefit of this code versus computing the Normal Equation directly using
# NumPy is that TensorFlow will automatically run this on your GPU card if you have
# one


#%% Implementing Gradient Descent
# Gradient Descent requires scaling the feature vectors first.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# Manually Computing the Gradients
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2 / m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

# Epoch 0 MSE = 9.161543
# Epoch 100 MSE = 0.7145006
# Epoch 200 MSE = 0.56670463
# Epoch 300 MSE = 0.5555716
# Epoch 400 MSE = 0.5488116
# Epoch 500 MSE = 0.5436362
# Epoch 600 MSE = 0.53962916
# Epoch 700 MSE = 0.5365092
# Epoch 800 MSE = 0.53406775
# Epoch 900 MSE = 0.53214705


# Using autodiff
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

gradients = tf.gradients(mse, [theta])[0]

training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

print("Best theta:")
print(best_theta)

# Epoch 0 MSE = 9.161543
# Epoch 100 MSE = 0.7145006
# Epoch 200 MSE = 0.56670463
# Epoch 300 MSE = 0.5555716
# Epoch 400 MSE = 0.5488117
# Epoch 500 MSE = 0.5436362
# Epoch 600 MSE = 0.53962916
# Epoch 700 MSE = 0.53650916
# Epoch 800 MSE = 0.5340678
# Epoch 900 MSE = 0.53214717
# Best theta:
# [[ 2.0685525 ]
#  [ 0.8874027 ]
#  [ 0.14401658]
#  [-0.34770882]
#  [ 0.36178368]
#  [ 0.00393811]
#  [-0.04269556]
#  [-0.6614528 ]
#  [-0.6375277 ]]


# Using an Optimizer
# But it gets even easier: it also provides
# a number of optimizers out of the box, including a Gradient Descent optimizer
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

print("Best theta:")
print(best_theta)


#%% Feeding Data to the Training Algorithm
# Placeholder nodes
reset_graph()

A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

print(B_val_1)
# [[6. 7. 8.]]
print(B_val_2)
# [[ 9. 10. 11.]
#  [12. 13. 14.]]

# Mini-batch Gradient Descent
# n_epochs = 1000
learning_rate = 0.01

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()


#%% Saving and Restoring Models
# Once you have trained your model, you should save its parameters to disk so you can
# come back to it whenever you want, use it in another program, compare it to other
# models, and so on
reset_graph()

n_epochs = 1000                                                                       # not shown in the book
learning_rate = 0.01                                                                  # not shown

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")            # not shown
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")            # not shown
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")                                      # not shown
error = y_pred - y                                                                    # not shown
mse = tf.reduce_mean(tf.square(error), name="mse")                                    # not shown
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)            # not shown
training_op = optimizer.minimize(mse)                                                 # not shown

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())  # not shown
            save_path = saver.save(sess, "D:/Documentos/MAESTRIA/3er cuatrimestre/Seminario Tensorflow/Libros/Ejemplos/data/my_model.ckpt")
        sess.run(training_op)

    best_theta = theta.eval()
    save_path = saver.save(sess, "D:/Documentos/MAESTRIA/3er cuatrimestre/Seminario Tensorflow/Libros/Ejemplos/data/my_model_final.ckpt")

# Restoring a model is just as easy: at the beginning of the execution phase,
# instead of initializing
# the variables using the init node, you call the restore() method of the
# Saver object:
with tf.Session() as sess:
    saver.restore(sess, "D:/Documentos/MAESTRIA/3er cuatrimestre/Seminario Tensorflow/Libros/Ejemplos/data/my_model_final.ckpt")
    best_theta_restored = theta.eval() # not shown in the book

np.allclose(best_theta, best_theta_restored)
# Out[33]: True

# If you want to have a saver that loads and restores theta with a different name,
# such as "weights":
saver = tf.train.Saver({"weights": theta})

# By default the saver also saves the graph structure itself in a second file
# with the extension .meta. You can use the function tf.train.import_meta_graph()
# to restore the graph structure.
# This function loads the graph into the default graph and returns a Saver
# that can then be used to restore the graph state (i.e., the variable values):
reset_graph()
# notice that we start with an empty graph.

saver = tf.train.import_meta_graph("D:/Documentos/MAESTRIA/3er cuatrimestre/Seminario Tensorflow/Libros/Ejemplos/data/my_model_final.ckpt.meta")  # this loads the graph structure
theta = tf.get_default_graph().get_tensor_by_name("theta:0") # not shown in the book

with tf.Session() as sess:
    saver.restore(sess, "D:/Documentos/MAESTRIA/3er cuatrimestre/Seminario Tensorflow/Libros/Ejemplos/data/my_model_final.ckpt")  # this restores the graph's state
    best_theta_restored = theta.eval() # not shown in the book

np.allclose(best_theta, best_theta_restored)
# Out[37]: True


#%% Visualizing the Graph and Training Curves Using TensorBoard
reset_graph()

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:                                                        # not shown in the book
    sess.run(init)                                                                # not shown

    for epoch in range(n_epochs):                                                 # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.close()

# Now it’s time to fire up the TensorBoard server
# tensorboard --logdir=D:\tf_logs


#%% Name Scopes

reset_graph()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.flush()
file_writer.close()


# Modularity
reset_graph()

def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)                          # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")    # not shown
        b = tf.Variable(0.0, name="bias")                             # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                      # not shown
        return tf.maximum(z, 0., name="max")                          # not shown

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())
file_writer.close()


# Sharing a threshold variable the classic way,
# by defining it outside of the relu() function then passing it as a parameter:
reset_graph()

def relu(X, threshold):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)                        # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, threshold, name="max")

threshold = tf.Variable(0.0, name="threshold")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name="output")

# Yet another option is to set the shared variable
# as an attribute of the relu() function upon the first call, like so:
reset_graph()

def relu(X):
    with tf.name_scope("relu"):
        if not hasattr(relu, "threshold"):
            relu.threshold = tf.Variable(0.0, name="threshold")
        w_shape = int(X.get_shape()[1]), 1                          # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, relu.threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")


# The
# idea is to use the get_variable() function to create the shared variable if it does not
# exist yet, or reuse it if it already exists. The desired behavior (creating or reusing) is
# controlled by an attribute of the current variable_scope(). For example, the following
# code will create a variable named "relu/threshold" (as a scalar, since shape=(),
# and using 0.0 as the initial value):
reset_graph()

with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))

with tf.variable_scope("relu", reuse=True):
    threshold = tf.get_variable("threshold")

with tf.variable_scope("relu") as scope:
    scope.reuse_variables()
    threshold = tf.get_variable("threshold")

reset_graph()

def relu(X):
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold")
        w_shape = int(X.get_shape()[1]), 1                          # not shown
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph())
file_writer.close()


# the following
# code creates the threshold variable within the relu() function upon the first call,
# then reuses it in subsequent calls
reset_graph()

def relu(X):
    with tf.variable_scope("relu"):
        threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("", default_name="") as scope:
    first_relu = relu(X)     # create the shared variable
    scope.reuse_variables()  # then reuse it
    relus = [first_relu] + [relu(X) for i in range(4)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu8", tf.get_default_graph())
file_writer.close()


