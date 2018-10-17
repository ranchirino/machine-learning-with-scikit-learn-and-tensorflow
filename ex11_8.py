# 8.1 Exercise: Build a DNN with five hidden layers of 100 neurons each, He initialization,
# and the ELU activation function.

# We will need similar DNNs in the next exercises, so let's create a function to build this DNN:

import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

he_init = tf.variance_scaling_initializer()

def dnn(inputs, n_hidden_layers=5, n_neurons=100, name=None, activation=tf.nn.elu, initializer=he_init):
    with tf.variable_scope(name, "dnn"):
        for layer in range(n_hidden_layers):
            inputs = tf.layers.dense(inputs, n_neurons, activation=activation,
                                     kernel_initializer=initializer,
                                     name="hidden%d" % (layer + 1))
        return inputs

n_inputs = 28 * 28 # MNIST
n_outputs = 5

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

dnn_outputs = dnn(X)

logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
Y_proba = tf.nn.softmax(logits, name="Y_proba")

# 8.2 Exercise: Using Adam optimization and early stopping,
# try training it on MNIST but only on digits 0 to 4,
# as we will use transfer learning for digits 5 to 9 in the next exercise.
# You will need a softmax output layer with five neurons,
# and as always make sure to save checkpoints at regular intervals
# and save the final model so you can reuse it later.

# Let's complete the graph with the cost function, the training op, and all the other usual components:

learning_rate = 0.01

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

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

# Now let's create the training set, validation and test set
# (we need the validation set to implement early stopping):

X_train1 = X_train[y_train < 5]
y_train1 = y_train[y_train < 5]
X_valid1 = X_valid[y_valid < 5]
y_valid1 = y_valid[y_valid < 5]
X_test1 = X_test[y_test < 5]
y_test1 = y_test[y_test < 5]

n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train1))
        for rnd_indices in np.array_split(rnd_idx, len(X_train1) // batch_size):
            X_batch, y_batch = X_train1[rnd_indices], y_train1[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid1, y: y_valid1})
        if loss_val < best_loss:
            save_path = saver.save(sess, "./my_mnist_model_0_to_4.ckpt")
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
    saver.restore(sess, "./my_mnist_model_0_to_4.ckpt")
    acc_test = accuracy.eval(feed_dict={X: X_test1, y: y_test1})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

# 0	Validation loss: 0.204374	Best loss: 0.204374	Accuracy: 95.97%
# 1	Validation loss: 0.116461	Best loss: 0.116461	Accuracy: 97.65%
# 2	Validation loss: 1.217414	Best loss: 0.116461	Accuracy: 39.68%
# 3	Validation loss: 1.676107	Best loss: 0.116461	Accuracy: 38.86%
# 4	Validation loss: 1.668141	Best loss: 0.116461	Accuracy: 22.01%
# 5	Validation loss: 1.645146	Best loss: 0.116461	Accuracy: 22.01%
# 6	Validation loss: 1.679659	Best loss: 0.116461	Accuracy: 18.73%
# 7	Validation loss: 1.772852	Best loss: 0.116461	Accuracy: 22.01%
# 8	Validation loss: 1.699114	Best loss: 0.116461	Accuracy: 19.27%
# 9	Validation loss: 1.765483	Best loss: 0.116461	Accuracy: 20.91%
# 10	Validation loss: 1.629255	Best loss: 0.116461	Accuracy: 22.01%
# 11	Validation loss: 1.812789	Best loss: 0.116461	Accuracy: 22.01%
# 12	Validation loss: 1.675878	Best loss: 0.116461	Accuracy: 18.73%
# 13	Validation loss: 1.633258	Best loss: 0.116461	Accuracy: 20.91%
# 14	Validation loss: 1.652902	Best loss: 0.116461	Accuracy: 20.91%
# 15	Validation loss: 1.635943	Best loss: 0.116461	Accuracy: 20.91%
# 16	Validation loss: 1.718915	Best loss: 0.116461	Accuracy: 19.08%
# 17	Validation loss: 1.682458	Best loss: 0.116461	Accuracy: 19.27%
# 18	Validation loss: 1.675366	Best loss: 0.116461	Accuracy: 18.73%
# 19	Validation loss: 1.645805	Best loss: 0.116461	Accuracy: 19.08%
# 20	Validation loss: 1.722337	Best loss: 0.116461	Accuracy: 22.01%
# 21	Validation loss: 1.656422	Best loss: 0.116461	Accuracy: 22.01%
# Early stopping!
# INFO:tensorflow:Restoring parameters from ./my_mnist_model_0_to_4.ckpt
# Final test accuracy: 98.11%

# We get 98.11% accuracy on the test set.
# That's not too bad, but let's see if we can do better by tuning the hyperparameters.

# 8.3 Exercise: Tune the hyperparameters using cross-validation and see what precision you can achieve.

# Let's create a DNNClassifier class, compatible with Scikit-Learn's RandomizedSearchCV class, to perform hyperparameter tuning. Here are the key points of this implementation:
#
#     the __init__() method (constructor) does nothing more than create instance variables for each of the hyperparameters.
#     the fit() method creates the graph, starts a session and trains the model:
#         it calls the _build_graph() method to build the graph (much lile the graph we defined earlier). Once this method is done creating the graph, it saves all the important operations as instance variables for easy access by other methods.
#         the _dnn() method builds the hidden layers, just like the dnn() function above, but also with support for batch normalization and dropout (for the next exercises).
#         if the fit() method is given a validation set (X_valid and y_valid), then it implements early stopping. This implementation does not save the best model to disk, but rather to memory: it uses the _get_model_params() method to get all the graph's variables and their values, and the _restore_model_params() method to restore the variable values (of the best model found). This trick helps speed up training.
#         After the fit() method has finished training the model, it keeps the session open so that predictions can be made quickly, without having to save a model to disk and restore it for every prediction. You can close the session by calling the close_session() method.
#     the predict_proba() method uses the trained model to predict the class probabilities.
#     the predict() method calls predict_proba() and returns the class with the highest probability, for each instance.

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

# Let's see if we get the exact same accuracy as earlier using this class (without dropout or batch norm):

dnn_clf = DNNClassifier(random_state=42)
dnn_clf.fit(X_train1, y_train1, n_epochs=1000, X_valid=X_valid1, y_valid=y_valid1)

# 0	Validation loss: 0.204374	Best loss: 0.204374	Accuracy: 95.97%
# 1	Validation loss: 0.116461	Best loss: 0.116461	Accuracy: 97.65%
# 2	Validation loss: 1.217414	Best loss: 0.116461	Accuracy: 39.68%
# 3	Validation loss: 1.676107	Best loss: 0.116461	Accuracy: 38.86%
# 4	Validation loss: 1.668141	Best loss: 0.116461	Accuracy: 22.01%
# 5	Validation loss: 1.645146	Best loss: 0.116461	Accuracy: 22.01%
# 6	Validation loss: 1.679659	Best loss: 0.116461	Accuracy: 18.73%
# 7	Validation loss: 1.772852	Best loss: 0.116461	Accuracy: 22.01%
# 8	Validation loss: 1.699114	Best loss: 0.116461	Accuracy: 19.27%
# 9	Validation loss: 1.765483	Best loss: 0.116461	Accuracy: 20.91%
# 10	Validation loss: 1.629255	Best loss: 0.116461	Accuracy: 22.01%
# 11	Validation loss: 1.812789	Best loss: 0.116461	Accuracy: 22.01%
# 12	Validation loss: 1.675878	Best loss: 0.116461	Accuracy: 18.73%
# 13	Validation loss: 1.633258	Best loss: 0.116461	Accuracy: 20.91%
# 14	Validation loss: 1.652902	Best loss: 0.116461	Accuracy: 20.91%
# 15	Validation loss: 1.635943	Best loss: 0.116461	Accuracy: 20.91%
# 16	Validation loss: 1.718915	Best loss: 0.116461	Accuracy: 19.08%
# 17	Validation loss: 1.682458	Best loss: 0.116461	Accuracy: 19.27%
# 18	Validation loss: 1.675366	Best loss: 0.116461	Accuracy: 18.73%
# 19	Validation loss: 1.645805	Best loss: 0.116461	Accuracy: 19.08%
# 20	Validation loss: 1.722337	Best loss: 0.116461	Accuracy: 22.01%
# 21	Validation loss: 1.656422	Best loss: 0.116461	Accuracy: 22.01%
# 22	Validation loss: 1.643527	Best loss: 0.116461	Accuracy: 18.73%
# Early stopping!
# Out[7]:
# DNNClassifier(activation=<function elu at 0x0000017D889A2268>,
#        batch_norm_momentum=None, batch_size=20, dropout_rate=None,
#        initializer=<tensorflow.python.ops.init_ops.VarianceScaling object at 0x0000017D86FA9550>,
#        learning_rate=0.01, n_hidden_layers=5, n_neurons=100,
#        optimizer_class=<class 'tensorflow.python.training.adam.AdamOptimizer'>,
#        random_state=42)

# The model is trained, let's see if it gets the same accuracy as earlier:

from sklearn.metrics import accuracy_score

y_pred = dnn_clf.predict(X_test1)
accuracy_score(y_test1, y_pred)
# Out[8]: 0.9811247324382175

# Yep! Working fine.
# Now we can use Scikit-Learn's RandomizedSearchCV class to search for better hyperparameters
# (this may take over an hour, depending on your system):

from sklearn.model_selection import RandomizedSearchCV

def leaky_relu(alpha=0.01):
    def parametrized_leaky_relu(z, name=None):
        return tf.maximum(alpha * z, z, name=name)
    return parametrized_leaky_relu

param_distribs = {
    "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
    "batch_size": [10, 50, 100, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01), leaky_relu(alpha=0.1)],
    # you could also try exploring different numbers of hidden layers, different optimizers, etc.
    #"n_hidden_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #"optimizer_class": [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer, momentum=0.95)],
}

rnd_search = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,
                                random_state=42, verbose=2)
rnd_search.fit(X_train1, y_train1, X_valid=X_valid1, y_valid=y_valid1, n_epochs=1000)

# If you have Scikit-Learn 0.18 or earlier, you should upgrade, or use the fit_params argument:
# fit_params={"X_valid": X_valid1, "y_valid": y_valid1, "n_epochs": 1000}
# rnd_search = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,
#                                 fit_params=fit_params, random_state=42, verbose=2)
# rnd_search.fit(X_train1, y_train1)

rnd_search.best_params_
# Out[2]:
# {'activation': <function tensorflow.python.ops.gen_nn_ops.elu>,
#  'batch_size': 500,
#  'learning_rate': 0.01,
#  'n_neurons': 140}

y_pred = rnd_search.predict(X_test1)
accuracy_score(y_test1, y_pred)
# Out[3]: 0.9881299863786729

# Wonderful!
# It's a good idea to save this model:

rnd_search.best_estimator_.save("./my_best_mnist_model_0_to_4")

# restore_saver = tf.train.import_meta_graph("./my_best_mnist_model_0_to_4.meta")
# with tf.Session() as sess:
#     restore_saver.restore(sess, "./my_best_mnist_model_0_to_4")

# 8.4 Exercise: Now try adding Batch Normalization and compare the learning curves:
# is it converging faster than before? Does it produce a better model?

# Let's train the best model found, once again, to see how fast it converges
# (alternatively, you could tweak the code above to make it write summaries for TensorBoard,
# so you can visualize the learning curve):

dnn_clf = DNNClassifier(activation=leaky_relu(alpha=0.1), batch_size=500, learning_rate=0.01,
                        n_neurons=140, random_state=42)
dnn_clf.fit(X_train1, y_train1, n_epochs=1000, X_valid=X_valid1, y_valid=y_valid1)

# 0	Validation loss: 0.081921	Best loss: 0.081921	Accuracy: 97.46%
# 1	Validation loss: 0.050916	Best loss: 0.050916	Accuracy: 98.55%
# 2	Validation loss: 0.047901	Best loss: 0.047901	Accuracy: 98.63%
# 3	Validation loss: 0.051151	Best loss: 0.047901	Accuracy: 98.51%
# 4	Validation loss: 0.046926	Best loss: 0.046926	Accuracy: 98.63%
# 5	Validation loss: 0.036573	Best loss: 0.036573	Accuracy: 98.98%
# 6	Validation loss: 0.045715	Best loss: 0.036573	Accuracy: 98.75%
# 7	Validation loss: 0.049617	Best loss: 0.036573	Accuracy: 98.51%
# 8	Validation loss: 0.033907	Best loss: 0.033907	Accuracy: 98.79%
# 9	Validation loss: 0.065576	Best loss: 0.033907	Accuracy: 98.36%
# 10	Validation loss: 0.048339	Best loss: 0.033907	Accuracy: 98.79%
# 11	Validation loss: 0.043099	Best loss: 0.033907	Accuracy: 98.94%
# 12	Validation loss: 0.050423	Best loss: 0.033907	Accuracy: 98.91%
# 13	Validation loss: 0.050776	Best loss: 0.033907	Accuracy: 98.94%
# 14	Validation loss: 0.047902	Best loss: 0.033907	Accuracy: 98.91%
# 15	Validation loss: 0.039905	Best loss: 0.033907	Accuracy: 99.10%
# 16	Validation loss: 0.057555	Best loss: 0.033907	Accuracy: 98.94%
# 17	Validation loss: 0.063678	Best loss: 0.033907	Accuracy: 98.98%
# 18	Validation loss: 293.542450	Best loss: 0.033907	Accuracy: 47.19%
# 19	Validation loss: 1.123482	Best loss: 0.033907	Accuracy: 88.27%
# 20	Validation loss: 0.631501	Best loss: 0.033907	Accuracy: 89.41%
# 21	Validation loss: 0.234845	Best loss: 0.033907	Accuracy: 95.66%
# 22	Validation loss: 0.195577	Best loss: 0.033907	Accuracy: 96.05%
# 23	Validation loss: 0.218874	Best loss: 0.033907	Accuracy: 94.84%
# 24	Validation loss: 0.163145	Best loss: 0.033907	Accuracy: 95.66%
# 25	Validation loss: 0.129676	Best loss: 0.033907	Accuracy: 96.60%
# 26	Validation loss: 0.105474	Best loss: 0.033907	Accuracy: 97.34%
# 27	Validation loss: 0.181767	Best loss: 0.033907	Accuracy: 95.31%
# 28	Validation loss: 0.114946	Best loss: 0.033907	Accuracy: 96.83%
# 29	Validation loss: 0.108520	Best loss: 0.033907	Accuracy: 97.38%
# Early stopping!
# Out[5]:
# DNNClassifier(activation=<function leaky_relu.<locals>.parametrized_leaky_relu at 0x0000022C8EE8C6A8>,
#        batch_norm_momentum=None, batch_size=500, dropout_rate=None,
#        initializer=<tensorflow.python.ops.init_ops.VarianceScaling object at 0x0000022CE93A9860>,
#        learning_rate=0.01, n_hidden_layers=5, n_neurons=140,
#        optimizer_class=<class 'tensorflow.python.training.adam.AdamOptimizer'>,
#        random_state=42)

y_pred = dnn_clf.predict(X_test1)
accuracy_score(y_test1, y_pred)
# Out[6]: 0.9920217941233703

# Good, now let's use the exact same model, but this time with batch normalization:

dnn_clf_bn = DNNClassifier(activation=leaky_relu(alpha=0.1), batch_size=500, learning_rate=0.01,
                           n_neurons=90, random_state=42,
                           batch_norm_momentum=0.95)
dnn_clf_bn.fit(X_train1, y_train1, n_epochs=1000, X_valid=X_valid1, y_valid=y_valid1)

# 0	Validation loss: 0.042938	Best loss: 0.042938	Accuracy: 98.83%
# 1	Validation loss: 0.041960	Best loss: 0.041960	Accuracy: 98.51%
# 2	Validation loss: 0.035264	Best loss: 0.035264	Accuracy: 98.83%
# 3	Validation loss: 0.045348	Best loss: 0.035264	Accuracy: 98.83%
# 4	Validation loss: 0.034566	Best loss: 0.034566	Accuracy: 98.98%
# 5	Validation loss: 0.040601	Best loss: 0.034566	Accuracy: 98.75%
# 6	Validation loss: 0.037512	Best loss: 0.034566	Accuracy: 99.14%
# 7	Validation loss: 0.030123	Best loss: 0.030123	Accuracy: 99.30%
# 8	Validation loss: 0.037433	Best loss: 0.030123	Accuracy: 99.18%
# 9	Validation loss: 0.030020	Best loss: 0.030020	Accuracy: 99.22%
# 10	Validation loss: 0.030447	Best loss: 0.030020	Accuracy: 99.10%
# 11	Validation loss: 0.030182	Best loss: 0.030020	Accuracy: 99.06%
# 12	Validation loss: 0.042467	Best loss: 0.030020	Accuracy: 99.02%
# 13	Validation loss: 0.032052	Best loss: 0.030020	Accuracy: 99.22%
# 14	Validation loss: 0.033146	Best loss: 0.030020	Accuracy: 99.02%
# 15	Validation loss: 0.032366	Best loss: 0.030020	Accuracy: 99.14%
# 16	Validation loss: 0.024838	Best loss: 0.024838	Accuracy: 99.26%
# 17	Validation loss: 0.055703	Best loss: 0.024838	Accuracy: 98.91%
# 18	Validation loss: 0.028903	Best loss: 0.024838	Accuracy: 99.26%
# 19	Validation loss: 0.028258	Best loss: 0.024838	Accuracy: 99.26%
# 20	Validation loss: 0.029061	Best loss: 0.024838	Accuracy: 99.37%
# 21	Validation loss: 0.029094	Best loss: 0.024838	Accuracy: 99.26%
# 22	Validation loss: 0.036263	Best loss: 0.024838	Accuracy: 99.37%
# 23	Validation loss: 0.028187	Best loss: 0.024838	Accuracy: 99.45%
# 24	Validation loss: 0.026727	Best loss: 0.024838	Accuracy: 99.41%
# 25	Validation loss: 0.033164	Best loss: 0.024838	Accuracy: 99.37%
# 26	Validation loss: 0.046121	Best loss: 0.024838	Accuracy: 98.91%
# 27	Validation loss: 0.040506	Best loss: 0.024838	Accuracy: 99.14%
# 28	Validation loss: 0.043349	Best loss: 0.024838	Accuracy: 98.91%
# 29	Validation loss: 0.033916	Best loss: 0.024838	Accuracy: 99.41%
# 30	Validation loss: 0.045895	Best loss: 0.024838	Accuracy: 99.14%
# 31	Validation loss: 0.039530	Best loss: 0.024838	Accuracy: 99.37%
# 32	Validation loss: 0.047609	Best loss: 0.024838	Accuracy: 99.06%
# 33	Validation loss: 0.066468	Best loss: 0.024838	Accuracy: 98.59%
# 34	Validation loss: 0.038734	Best loss: 0.024838	Accuracy: 99.10%
# 35	Validation loss: 0.036475	Best loss: 0.024838	Accuracy: 99.18%
# 36	Validation loss: 0.035985	Best loss: 0.024838	Accuracy: 99.30%
# 37	Validation loss: 0.040642	Best loss: 0.024838	Accuracy: 99.14%
# Early stopping!
# Out[7]:
# DNNClassifier(activation=<function leaky_relu.<locals>.parametrized_leaky_relu at 0x0000022C8EE8C598>,
#        batch_norm_momentum=0.95, batch_size=500, dropout_rate=None,
#        initializer=<tensorflow.python.ops.init_ops.VarianceScaling object at 0x0000022CE93A9860>,
#        learning_rate=0.01, n_hidden_layers=5, n_neurons=90,
#        optimizer_class=<class 'tensorflow.python.training.adam.AdamOptimizer'>,
#        random_state=42)

y_pred = dnn_clf_bn.predict(X_test1)
accuracy_score(y_test1, y_pred)
# Out[8]: 0.9935785172212492

# # Let's see if we can find a good set of hyperparameters that will work well with batch normalization:
#
# param_distribs = {
#     "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
#     "batch_size": [10, 50, 100, 500],
#     "learning_rate": [0.01, 0.02, 0.05, 0.1],
#     "activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01), leaky_relu(alpha=0.1)],
#     # you could also try exploring different numbers of hidden layers, different optimizers, etc.
#     #"n_hidden_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     #"optimizer_class": [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer, momentum=0.95)],
#     "batch_norm_momentum": [0.9, 0.95, 0.98, 0.99, 0.999],
# }
#
# rnd_search_bn = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,
#                                    fit_params={"X_valid": X_valid1, "y_valid": y_valid1, "n_epochs": 1000},
#                                    random_state=42, verbose=2)
# rnd_search_bn.fit(X_train1, y_train1)
#
# # Esto demora mucho

# 8.5 Exercise: is the model overfitting the training set?
# Try adding dropout to every layer and try again. Does it help?

# Let's go back to the best model we trained earlier and see how it performs on the training set:
y_pred = dnn_clf.predict(X_train1)
accuracy_score(y_train1, y_pred)
# Out[15]: 0.9963977459162565

# The model performs significantly better on the training set than on the test set,
# which means it is overfitting the training set. A bit of regularization may help.
# Let's try adding dropout with a 50% dropout rate:

dnn_clf_dropout = DNNClassifier(activation=leaky_relu(alpha=0.1), batch_size=500, learning_rate=0.01,
                                n_neurons=90, random_state=42,
                                dropout_rate=0.5)
dnn_clf_dropout.fit(X_train1, y_train1, n_epochs=1000, X_valid=X_valid1, y_valid=y_valid1)

# 0	Validation loss: 0.116464	Best loss: 0.116464	Accuracy: 96.87%
# 1	Validation loss: 0.093813	Best loss: 0.093813	Accuracy: 97.62%
# 2	Validation loss: 0.082271	Best loss: 0.082271	Accuracy: 97.58%
# 3	Validation loss: 0.080310	Best loss: 0.080310	Accuracy: 97.97%
# 4	Validation loss: 0.085542	Best loss: 0.080310	Accuracy: 98.16%
# 5	Validation loss: 0.083293	Best loss: 0.080310	Accuracy: 98.01%
# 6	Validation loss: 0.067854	Best loss: 0.067854	Accuracy: 98.20%
# 7	Validation loss: 0.077205	Best loss: 0.067854	Accuracy: 98.16%
# 8	Validation loss: 0.065659	Best loss: 0.065659	Accuracy: 98.32%
# 9	Validation loss: 0.071699	Best loss: 0.065659	Accuracy: 98.20%
# 10	Validation loss: 0.069561	Best loss: 0.065659	Accuracy: 98.01%
# 11	Validation loss: 0.068030	Best loss: 0.065659	Accuracy: 98.44%
# 12	Validation loss: 0.070705	Best loss: 0.065659	Accuracy: 98.01%
# 13	Validation loss: 0.069951	Best loss: 0.065659	Accuracy: 98.20%
# 14	Validation loss: 0.072351	Best loss: 0.065659	Accuracy: 98.24%
# 15	Validation loss: 0.069938	Best loss: 0.065659	Accuracy: 98.12%
# 16	Validation loss: 0.070541	Best loss: 0.065659	Accuracy: 98.32%
# 17	Validation loss: 0.065904	Best loss: 0.065659	Accuracy: 98.40%
# 18	Validation loss: 0.066342	Best loss: 0.065659	Accuracy: 98.28%
# 19	Validation loss: 0.064135	Best loss: 0.064135	Accuracy: 98.40%
# 20	Validation loss: 0.065433	Best loss: 0.064135	Accuracy: 98.24%
# 21	Validation loss: 0.070290	Best loss: 0.064135	Accuracy: 98.05%
# 22	Validation loss: 0.089318	Best loss: 0.064135	Accuracy: 98.01%
# 23	Validation loss: 0.076181	Best loss: 0.064135	Accuracy: 97.97%
# 24	Validation loss: 0.068750	Best loss: 0.064135	Accuracy: 97.97%
# 25	Validation loss: 0.129305	Best loss: 0.064135	Accuracy: 97.15%
# 26	Validation loss: 0.130173	Best loss: 0.064135	Accuracy: 96.13%
# 27	Validation loss: 0.133423	Best loss: 0.064135	Accuracy: 95.97%
# 28	Validation loss: 0.139302	Best loss: 0.064135	Accuracy: 97.07%
# 29	Validation loss: 0.138757	Best loss: 0.064135	Accuracy: 96.76%
# 30	Validation loss: 0.125138	Best loss: 0.064135	Accuracy: 96.72%
# 31	Validation loss: 0.114705	Best loss: 0.064135	Accuracy: 97.69%
# 32	Validation loss: 0.096743	Best loss: 0.064135	Accuracy: 97.89%
# 33	Validation loss: 0.095120	Best loss: 0.064135	Accuracy: 97.85%
# 34	Validation loss: 0.089346	Best loss: 0.064135	Accuracy: 97.97%
# 35	Validation loss: 0.079778	Best loss: 0.064135	Accuracy: 98.05%
# 36	Validation loss: 0.085033	Best loss: 0.064135	Accuracy: 98.16%
# 37	Validation loss: 0.080816	Best loss: 0.064135	Accuracy: 98.01%
# 38	Validation loss: 0.090679	Best loss: 0.064135	Accuracy: 98.16%
# 39	Validation loss: 0.074215	Best loss: 0.064135	Accuracy: 98.28%
# 40	Validation loss: 0.070389	Best loss: 0.064135	Accuracy: 98.28%
# Early stopping!
# Out[16]:
# DNNClassifier(activation=<function leaky_relu.<locals>.parametrized_leaky_relu at 0x000002449A766E18>,
#        batch_norm_momentum=None, batch_size=500, dropout_rate=0.5,
#        initializer=<tensorflow.python.ops.init_ops.VarianceScaling object at 0x0000024490B267B8>,
#        learning_rate=0.01, n_hidden_layers=5, n_neurons=90,
#        optimizer_class=<class 'tensorflow.python.training.adam.AdamOptimizer'>,
#        random_state=42)

# Let's check the accuracy:
y_pred = dnn_clf_dropout.predict(X_test1)
accuracy_score(y_test1, y_pred)
# Out[17]: 0.9863786728935591

# We are out of luck, dropout does not seem to help either.
# Let's try tuning the hyperparameters, perhaps we can squeeze a bit more performance out of this model:

# param_distribs = {
#     "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
#     "batch_size": [10, 50, 100, 500],
#     "learning_rate": [0.01, 0.02, 0.05, 0.1],
#     "activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01), leaky_relu(alpha=0.1)],
#     # you could also try exploring different numbers of hidden layers, different optimizers, etc.
#     #"n_hidden_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     #"optimizer_class": [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer, momentum=0.95)],
#     "dropout_rate": [0.2, 0.3, 0.4, 0.5, 0.6],
# }
#
# rnd_search_dropout = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,
#                                         fit_params={"X_valid": X_valid1, "y_valid": y_valid1, "n_epochs": 1000},
#                                         random_state=42, verbose=2)
# rnd_search_dropout.fit(X_train1, y_train1)

