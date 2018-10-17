# Transfer learning for large image classification

# 9.1.
# Exercise: Create a training set containing at least 100 images per class.
# For example, you could classify your own pictures based on the location (beach, mountain, city, etc.),
# or alternatively you can just use an existing dataset, such as the flowers dataset or MIT's places dataset (requires registration, and it is huge).

# Let's tackle the flowers dataset. First, we need to download it:

import numpy as np
import os
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

width = 299
height = 299
channels = 3

import sys
import tarfile
from six.moves import urllib

FLOWERS_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
FLOWERS_PATH = os.path.join("datasets", "flowers")

def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()

def fetch_flowers(url=FLOWERS_URL, path=FLOWERS_PATH):
    if os.path.exists(FLOWERS_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "flower_photos.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    flowers_tgz = tarfile.open(tgz_path)
    flowers_tgz.extractall(path=path)
    flowers_tgz.close()
    os.remove(tgz_path)

fetch_flowers()

# Each subdirectory of the flower_photos directory contains all the pictures of a given class. Let's get the list of classes:

flowers_root_path = os.path.join(FLOWERS_PATH, "flower_photos")
flower_classes = sorted([dirname for dirname in os.listdir(flowers_root_path)
                  if os.path.isdir(os.path.join(flowers_root_path, dirname))])
flower_classes
# Out[3]: ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Let's get the list of all the image file paths for each class:

from collections import defaultdict

image_paths = defaultdict(list)

for flower_class in flower_classes:
    image_dir = os.path.join(flowers_root_path, flower_class)
    for filepath in os.listdir(image_dir):
        if filepath.endswith(".jpg"):
            image_paths[flower_class].append(os.path.join(image_dir, filepath))

# Let's sort the image paths just to make this notebook behave consistently across multiple runs:

for paths in image_paths.values():
    paths.sort()

# Let's take a peek at the first few images from each class:

import matplotlib.image as mpimg

n_examples_per_class = 2

for flower_class in flower_classes:
    print("Class:", flower_class)
    plt.figure(figsize=(10,5))
    for index, example_image_path in enumerate(image_paths[flower_class][:n_examples_per_class]):
        example_image = mpimg.imread(example_image_path)[:, :, :channels]
        plt.subplot(100 + n_examples_per_class * 10 + index + 1)
        plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
        plt.imshow(example_image)
        plt.axis("off")
    plt.show()

# Class: daisy
# Class: dandelion
# Class: roses
# Class: sunflowers
# Class: tulips


# 9.2.
# Exercise: Write a preprocessing step that will resize and crop the image to 299 × 299, with some randomness for data augmentation.

# First, let's implement this using NumPy and SciPy:
#     using basic NumPy slicing for image cropping,
#     NumPy's fliplr() function to flip the image horizontally (with 50% probability),
#     and SciPy's imresize() function for zooming.
#         Note that imresize() is based on the Python Image Library (PIL).

from scipy.misc import imresize


def prepare_image(image, target_width=299, target_height=299, max_zoom=0.2):
    """Zooms and crops the image randomly for data augmentation."""

    # First, let's find the largest bounding box with the target size ratio that fits within the image
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height

    # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number
    # between 1.0 and 1.0 + `max_zoom`.
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)

    # Next, we can select a random location on the image for this bounding box.
    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    # Let's crop the image using the random bounding box we built.
    image = image[y0:y1, x0:x1]

    # Let's also flip the image horizontally with 50% probability:
    if np.random.rand() < 0.5:
        image = np.fliplr(image)

    # Now, let's resize the image to the target dimensions.
    image = imresize(image, (target_width, target_height))

    # Finally, let's ensure that the colors are represented as
    # 32-bit floats ranging from 0.0 to 1.0 (for now):
    return image.astype(np.float32) / 255

# Note: at test time, the preprocessing step should be as light as possible,
# just the bare minimum necessary to be able to feed the image to the neural network.
# You may want to tweak the above function to add a training parameter: if False,
# preprocessing should be limited to the bare minimum (i.e., no flipping the image,
# and just the minimum cropping required, preserving the center of the image).

# Let's check out the result on this image:

plt.figure(figsize=(6, 8))
plt.imshow(example_image)
plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
plt.axis("off")
plt.show()

# There we go:

prepared_image = prepare_image(example_image)

plt.figure(figsize=(8, 8))
plt.imshow(prepared_image)
plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
plt.axis("off")
plt.show()

# Now let's look at a few other random images generated from the same original image:

rows, cols = 2, 3

plt.figure(figsize=(14, 8))
for row in range(rows):
    for col in range(cols):
        prepared_image = prepare_image(example_image)
        plt.subplot(rows, cols, row * cols + col + 1)
        plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
        plt.imshow(prepared_image)
        plt.axis("off")
plt.show()



# Looks good!

# Alternatively, it's also possible to implement this image preprocessing step directly with TensorFlow,
# using the functions in the tf.image module (see the API for the full list).
# As you can see, this function looks very much like the one above, except it does not actually perform the image transformation,
# but rather creates a set of TensorFlow operations that will perform the transformation when you run the graph.

def prepare_image_with_tensorflow(image, target_width = 299, target_height = 299, max_zoom = 0.2):
    """Zooms and crops the image randomly for data augmentation."""

    # First, let's find the largest bounding box with the target size ratio that fits within the image
    image_shape = tf.cast(tf.shape(image), tf.float32)
    height = image_shape[0]
    width = image_shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = tf.cond(crop_vertically,
                         lambda: width,
                         lambda: height * target_image_ratio)
    crop_height = tf.cond(crop_vertically,
                          lambda: width / target_image_ratio,
                          lambda: height)

    # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number
    # between 1.0 and 1.0 + `max_zoom`.
    resize_factor = tf.random_uniform(shape=[], minval=1.0, maxval=1.0 + max_zoom)
    crop_width = tf.cast(crop_width / resize_factor, tf.int32)
    crop_height = tf.cast(crop_height / resize_factor, tf.int32)
    box_size = tf.stack([crop_height, crop_width, 3])   # 3 = number of channels

    # Let's crop the image using a random bounding box of the size we computed
    image = tf.random_crop(image, box_size)

    # Let's also flip the image horizontally with 50% probability:
    image = tf.image.random_flip_left_right(image)

    # The resize_bilinear function requires a 4D tensor (a batch of images)
    # so we need to expand the number of dimensions first:
    image_batch = tf.expand_dims(image, 0)

    # Finally, let's resize the image to the target dimensions. Note that this function
    # returns a float32 tensor.
    image_batch = tf.image.resize_bilinear(image_batch, [target_height, target_width])
    image = image_batch[0] / 255  # back to a single image, and scale the colors from 0.0 to 1.0
    return image


# Let's test this function!

reset_graph()

input_image = tf.placeholder(tf.uint8, shape=[None, None, 3])
prepared_image_op = prepare_image_with_tensorflow(input_image)

with tf.Session():
    prepared_image = prepared_image_op.eval(feed_dict={input_image: example_image})

plt.figure(figsize=(6, 6))
plt.imshow(prepared_image)
plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
plt.axis("off")
plt.show()

# Looks perfect!


# 9.3.
# Exercise: Using the pretrained Inception v3 model from the previous exercise,
# freeze all layers up to the bottleneck layer (i.e., the last layer before the output layer),
# and replace the output layer with the appropriate number of outputs for your new classification task
# (e.g., the flowers dataset has five mutually exclusive classes so the output layer must have five neurons and use the softmax activation function).

# Let's start by fetching the inception v3 graph again.
# This time, let's use a training placeholder that we will use to tell TensorFlow whether we are training the network or not
# (this is needed by operations such as dropout and batch normalization).

from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

reset_graph()

X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
training = tf.placeholder_with_default(False, shape=[])
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=training)

inception_saver = tf.train.Saver()

# Now we need to find the point in the graph where we should attach the new output layer.
# It should be the layer right before the current output layer.
# One way to do this is to explore the output layer's inputs:

logits.op.inputs[0]
# Out[22]: <tf.Tensor 'InceptionV3/Logits/Conv2d_1c_1x1/BiasAdd:0' shape=(?, 1, 1, 1001) dtype=float32>

# Nope, that's part of the output layer (adding the biases). Let's continue walking backwards in the graph:

logits.op.inputs[0].op.inputs[0]
# Out[23]: <tf.Tensor 'InceptionV3/Logits/Conv2d_1c_1x1/Conv2D:0' shape=(?, 1, 1, 1001) dtype=float32>

# That's also part of the output layer, it's the final layer in the inception layer
# (if you are not sure you can visualize the graph using TensorBoard). Once again, let's continue walking backwards in the graph:

logits.op.inputs[0].op.inputs[0].op.inputs[0]
# Out[24]: <tf.Tensor 'InceptionV3/Logits/Dropout_1b/cond/Merge:0' shape=(?, 1, 1, 2048) dtype=float32>

# Aha! There we are, this is the output of the dropout layer.
# This is the very last layer before the output layer in the Inception v3 network, so that's the layer we need to build upon.
# Note that there was actually a simpler way to find this layer: the inception_v3() function returns a dict of end points:

end_points
# Out[25]:
# {'Conv2d_1a_3x3': <tf.Tensor 'InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu:0' shape=(?, 149, 149, 32) dtype=float32>,
#  'Conv2d_2a_3x3': <tf.Tensor 'InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu:0' shape=(?, 147, 147, 32) dtype=float32>,
#  'Conv2d_2b_3x3': <tf.Tensor 'InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu:0' shape=(?, 147, 147, 64) dtype=float32>,
#  'MaxPool_3a_3x3': <tf.Tensor 'InceptionV3/InceptionV3/MaxPool_3a_3x3/MaxPool:0' shape=(?, 73, 73, 64) dtype=float32>,
#  'Conv2d_3b_1x1': <tf.Tensor 'InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu:0' shape=(?, 73, 73, 80) dtype=float32>,
#  'Conv2d_4a_3x3': <tf.Tensor 'InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu:0' shape=(?, 71, 71, 192) dtype=float32>,
#  'MaxPool_5a_3x3': <tf.Tensor 'InceptionV3/InceptionV3/MaxPool_5a_3x3/MaxPool:0' shape=(?, 35, 35, 192) dtype=float32>,
#  'Mixed_5b': <tf.Tensor 'InceptionV3/InceptionV3/Mixed_5b/concat:0' shape=(?, 35, 35, 256) dtype=float32>,
#  'Mixed_5c': <tf.Tensor 'InceptionV3/InceptionV3/Mixed_5c/concat:0' shape=(?, 35, 35, 288) dtype=float32>,
#  'Mixed_5d': <tf.Tensor 'InceptionV3/InceptionV3/Mixed_5d/concat:0' shape=(?, 35, 35, 288) dtype=float32>,
#  'Mixed_6a': <tf.Tensor 'InceptionV3/InceptionV3/Mixed_6a/concat:0' shape=(?, 17, 17, 768) dtype=float32>,
#  'Mixed_6b': <tf.Tensor 'InceptionV3/InceptionV3/Mixed_6b/concat:0' shape=(?, 17, 17, 768) dtype=float32>,
#  'Mixed_6c': <tf.Tensor 'InceptionV3/InceptionV3/Mixed_6c/concat:0' shape=(?, 17, 17, 768) dtype=float32>,
#  'Mixed_6d': <tf.Tensor 'InceptionV3/InceptionV3/Mixed_6d/concat:0' shape=(?, 17, 17, 768) dtype=float32>,
#  'Mixed_6e': <tf.Tensor 'InceptionV3/InceptionV3/Mixed_6e/concat:0' shape=(?, 17, 17, 768) dtype=float32>,
#  'Mixed_7a': <tf.Tensor 'InceptionV3/InceptionV3/Mixed_7a/concat:0' shape=(?, 8, 8, 1280) dtype=float32>,
#  'Mixed_7b': <tf.Tensor 'InceptionV3/InceptionV3/Mixed_7b/concat:0' shape=(?, 8, 8, 2048) dtype=float32>,
#  'Mixed_7c': <tf.Tensor 'InceptionV3/InceptionV3/Mixed_7c/concat:0' shape=(?, 8, 8, 2048) dtype=float32>,
#  'AuxLogits': <tf.Tensor 'InceptionV3/AuxLogits/SpatialSqueeze:0' shape=(?, 1001) dtype=float32>,
#  'PreLogits': <tf.Tensor 'InceptionV3/Logits/Dropout_1b/cond/Merge:0' shape=(?, 1, 1, 2048) dtype=float32>,
#  'Logits': <tf.Tensor 'InceptionV3/Logits/SpatialSqueeze:0' shape=(?, 1001) dtype=float32>,
#  'Predictions': <tf.Tensor 'InceptionV3/Predictions/Reshape_1:0' shape=(?, 1001) dtype=float32>}

# As you can see, the "PreLogits" end point is precisely what we need:

end_points["PreLogits"]
# Out[26]: <tf.Tensor 'InceptionV3/Logits/Dropout_1b/cond/Merge:0' shape=(?, 1, 1, 2048) dtype=float32>

# We can drop the 2nd and 3rd dimensions using the tf.squeeze() function:

prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])

# Then we can add the final fully connected layer on top of this layer:

n_outputs = len(flower_classes)

with tf.name_scope("new_output_layer"):
    flower_logits = tf.layers.dense(prelogits, n_outputs, name="flower_logits")
    Y_proba = tf.nn.softmax(flower_logits, name="Y_proba")

# Finally, we need to add the usual bits and pieces:
#
#     the placeholder for the targets (y),
#     the loss function, which is the cross-entropy, as usual for a classification task,
#     an optimizer, that we use to create a training operation that will minimize the cost function,
#     a couple operations to measure the model's accuracy,
#     and finally an initializer and a saver.
#
# There is one important detail, however: since we want to train only the output layer
# (all other layers must be frozen), we must pass the list of variables to train to the optimizer's minimize() method:

y = tf.placeholder(tf.int32, shape=[None])

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flower_logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    flower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="flower_logits")
    training_op = optimizer.minimize(loss, var_list=flower_vars)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(flower_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

[v.name for v in flower_vars]
# Out[30]: ['flower_logits/kernel:0', 'flower_logits/bias:0']

# Notice that we created the inception_saver before adding the new output layer:
# we will use this saver to restore the pretrained model state,
# so we don't want it to try to restore new variables (it would just fail saying it does not know the new variables).
# The second saver will be used to save the final flower model, including both the pretrained variables and the new ones.


# 9.4.
# Exercise: Split your dataset into a training set and a test set.
# Train the model on the training set and evaluate it on the test set.

# First, we will want to represent the classes as ints rather than strings:

flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes)}
flower_class_ids
# Out[31]: {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}

# It will be easier to shuffle the dataset set if we represent it as a list of filepath/class pairs:

flower_paths_and_classes = []
for flower_class, paths in image_paths.items():
    for path in paths:
        flower_paths_and_classes.append((path, flower_class_ids[flower_class]))

# Next, lets shuffle the dataset and split it into the training set and the test set:

test_ratio = 0.2
train_size = int(len(flower_paths_and_classes) * (1 - test_ratio))

np.random.shuffle(flower_paths_and_classes)

flower_paths_and_classes_train = flower_paths_and_classes[:train_size]
flower_paths_and_classes_test = flower_paths_and_classes[train_size:]

# Let's look at the first 3 instances in the training set:

flower_paths_and_classes_train[:3]
# Out[38]:
# [('datasets\\flowers\\flower_photos\\daisy\\19834392829_7d697871f6.jpg', 0),
#  ('datasets\\flowers\\flower_photos\\sunflowers\\5957007921_62333981d2_n.jpg', 3),
#  ('datasets\\flowers\\flower_photos\\tulips\\7166635566_ee240b5408_n.jpg', 4)]

# Next, we will also need a function to preprocess a set of images.
# This function will be useful to preprocess the test set, and also to create batches during training.
# For simplicity, we will use the NumPy/SciPy implementation:

from random import sample

def prepare_batch(flower_paths_and_classes, batch_size):
    batch_paths_and_classes = sample(flower_paths_and_classes, batch_size)
    images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch

X_batch, y_batch = prepare_batch(flower_paths_and_classes_train, batch_size=4)

print(X_batch.shape)
print(X_batch.dtype)
print(y_batch.shape)
print(y_batch.dtype)
# (4, 299, 299, 3)
# float32
# (4,)
# int32

# Looking good. Now let's use this function to prepare the test set:

X_test, y_test = prepare_batch(flower_paths_and_classes_test, batch_size=len(flower_paths_and_classes_test))

X_test.shape
# Out[44]: (734, 299, 299, 3)

# We could prepare the training set in much the same way, but it would only generate one variant for each image.
# Instead, it's preferable to generate the training batches on the fly during training,
# so that we can really benefit from data augmentation, with many variants of each image.

# And now, we are ready to train the network (or more precisely, the output layer we just added, since all the other layers are frozen).
# Be aware that this may take a (very) long time.

n_epochs = 10
batch_size = 40
n_iterations_per_epoch = len(flower_paths_and_classes_train) // batch_size
INCEPTION_PATH = os.path.join("datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")

with tf.Session() as sess:
    init.run()
    inception_saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)

    for epoch in range(n_epochs):
        print("Epoch", epoch, end="")
        for iteration in range(n_iterations_per_epoch):
            print(".", end="")
            X_batch, y_batch = prepare_batch(flower_paths_and_classes_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("  Train accuracy:", acc_train)

        save_path = saver.save(sess, "./my_flowers_model")

# Epoch 0.........................................................................  Train accuracy: 0.75
# Epoch 1.........................................................................  Train accuracy: 0.625
# Epoch 2.........................................................................  Train accuracy: 0.675
# Epoch 3.........................................................................  Train accuracy: 0.6
# Epoch 4.........................................................................  Train accuracy: 0.65
# Epoch 5.........................................................................  Train accuracy: 0.625
# Epoch 6.........................................................................  Train accuracy: 0.775
# Epoch 7.........................................................................  Train accuracy: 0.65
# Epoch 8.........................................................................  Train accuracy: 0.575
# Epoch 9.........................................................................  Train accuracy: 0.8

n_test_batches = 10
X_test_batches = np.array_split(X_test, n_test_batches)
y_test_batches = np.array_split(y_test, n_test_batches)

with tf.Session() as sess:
    saver.restore(sess, "./my_flowers_model")

    print("Computing final accuracy on the test set (this will take a while)...")
    acc_test = np.mean([
        accuracy.eval(feed_dict={X: X_test_batch, y: y_test_batch})
        for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)])
    print("Test accuracy:", acc_test)

# INFO:tensorflow:Restoring parameters from ./my_flowers_model
# Computing final accuracy on the test set (this will take a while)...
# Test accuracy: 0.64990747

