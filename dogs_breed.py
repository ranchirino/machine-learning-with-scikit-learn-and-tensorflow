import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

#%% Process the data
import tarfile

dogs_path = os.path.join("datasets", "dogs")
tar_name = "images.tar"

def fetch_data(path=dogs_path, file_name=tar_name):
    tar_path = os.path.join(path, file_name)
    tar = tarfile.open(tar_path)
    # tar_members = tar.getmembers()
    tar.extractall(path=path)
    tar.close()

# extract the images
fetch_data()

# get the breed name (classes)
full_dogs_path = os.path.join(dogs_path, "Images")
dogs_breeds = [dirname for dirname in os.listdir(full_dogs_path) if os.path.isdir(os.path.join(full_dogs_path, dirname))]

dog_classes = []
for breed in dogs_breeds:
    dog_classes.append(breed.split('-', maxsplit=1)[1].replace('_', ' ').replace('-', ' ').capitalize())

# dog_classes = sorted(dog_classes)

# let's get the classes with their image path
from collections import defaultdict

image_paths = defaultdict(list)

for index, breed in enumerate(dogs_breeds):
    image_dir = os.path.join(full_dogs_path, breed)
    for image in os.listdir(image_dir):
        if image.endswith(".jpg"):
            image_paths[dog_classes[index]].append(os.path.join(image_dir, image))

# let's see images of a few classes
import matplotlib.image as mpimg

n_examples_class = 5
n_examples_per_class = 2

for dog_class in dog_classes[:n_examples_class]:
    print("Class:", dog_class)
    plt.figure(figsize=(10,5))
    for index, example_image_path in enumerate(image_paths[dog_class][:n_examples_per_class]):
        example_image = mpimg.imread(example_image_path)
        plt.subplot(100 + n_examples_per_class * 10 + index + 1)
        plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
        plt.imshow(example_image)
        plt.axis("off")
    plt.show()

# preprocessing step that will resize and crop the image to 299 × 299, with some randomness for data augmentation.

from scipy.misc import imresize
from skimage.transform import resize

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

    # Now, let's resize the image to the target dimensions.
    image = imresize(image, (target_width, target_height))
    # image = resize(image, (target_width, target_height), anti_aliasing=True, mode='reflect')

    # Finally, let's ensure that the colors are represented as
    # 32-bit floats ranging from 0.0 to 1.0 (for now):
    return image.astype(np.float32) / 255

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

# Alternativamente, también es posible implementar este paso de preprocesamiento de imágenes directamente con TensorFlow,
# utilizando las funciones en el módulo tf.image (consulte la API para ver la lista completa). Como puede ver,
# esta función se parece mucho a la anterior, excepto que en realidad no realiza la transformación de la imagen,
# sino que crea un conjunto de operaciones TensorFlow que realizarán la transformación cuando ejecute el gráfico.

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


#%% Let's start by fetching the inception v3 graph again.
# This time, let's use a training placeholder that we will use to tell TensorFlow whether we are training the network or not
# (this is needed by operations such as dropout and batch normalization).

from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

reset_graph()

width = 299
height = 299
channels = 3

X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
training = tf.placeholder_with_default(False, shape=[])
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=training)

inception_saver = tf.train.Saver()

# Now we need to find the point in the graph where we should attach the new output layer.
# It should be the layer right before the current output layer.

end_points
# Out[7]:
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
# Out[8]: <tf.Tensor 'InceptionV3/Logits/Dropout_1b/cond/Merge:0' shape=(?, 1, 1, 2048) dtype=float32>

# We can drop the 2nd and 3rd dimensions using the tf.squeeze() function:
prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])

# Then we can add the final fully connected layer on top of this layer:
n_outputs = len(dog_classes)
n_outputs
# Out[13]: 120

with tf.name_scope("new_output_layer"):
    dog_logits = tf.layers.dense(prelogits, n_outputs, name="dog_logits")
    Y_proba = tf.nn.softmax(dog_logits, name="Y_proba")

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
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dog_logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    dog_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dog_logits")
    training_op = optimizer.minimize(loss, var_list=dog_vars)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(dog_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

[v.name for v in dog_vars]
# Out[17]: ['dog_logits/kernel:0', 'dog_logits/bias:0']

# Notice that we created the inception_saver before adding the new output layer:
# we will use this saver to restore the pretrained model state,
# so we don't want it to try to restore new variables (it would just fail saying it does not know the new variables).
# The second saver will be used to save the final flower model, including both the pretrained variables and the new ones.


#%% get a trainig set and a test set
# we will want to represent the classes as ints rather than strings
dog_class_ids = {dog_class: index for index, dog_class in enumerate(dog_classes)}
dog_class_ids
# Out[18]:
# {'Chihuahua': 0,
#  'Japanese spaniel': 1,
#  'Maltese dog': 2,
#  'Pekinese': 3,
#  'Shih tzu': 4,
#  'Blenheim spaniel': 5,
#  'Papillon': 6,
#  'Toy terrier': 7,
#  'Rhodesian ridgeback': 8,
#  'Afghan hound': 9,
#  'Basset': 10,
#  'Beagle': 11,
#  'Bloodhound': 12,
#  'Bluetick': 13,
#  'Black and tan coonhound': 14,
#  'Walker hound': 15,
#  'English foxhound': 16,
#  'Redbone': 17,
#  'Borzoi': 18,
#  'Irish wolfhound': 19,
#  'Italian greyhound': 20,
#  'Whippet': 21,
#  'Ibizan hound': 22,
#  'Norwegian elkhound': 23,
#  'Otterhound': 24,
#  'Saluki': 25,
#  'Scottish deerhound': 26,
#  'Weimaraner': 27,
#  'Staffordshire bullterrier': 28,
#  'American staffordshire terrier': 29,
#  'Bedlington terrier': 30,
#  'Border terrier': 31,
#  'Kerry blue terrier': 32,
#  'Irish terrier': 33,
#  'Norfolk terrier': 34,
#  'Norwich terrier': 35,
#  'Yorkshire terrier': 36,
#  'Wire haired fox terrier': 37,
#  'Lakeland terrier': 38,
#  'Sealyham terrier': 39,
#  'Airedale': 40,
#  'Cairn': 41,
#  'Australian terrier': 42,
#  'Dandie dinmont': 43,
#  'Boston bull': 44,
#  'Miniature schnauzer': 45,
#  'Giant schnauzer': 46,
#  'Standard schnauzer': 47,
#  'Scotch terrier': 48,
#  'Tibetan terrier': 49,
#  'Silky terrier': 50,
#  'Soft coated wheaten terrier': 51,
#  'West highland white terrier': 52,
#  'Lhasa': 53,
#  'Flat coated retriever': 54,
#  'Curly coated retriever': 55,
#  'Golden retriever': 56,
#  'Labrador retriever': 57,
#  'Chesapeake bay retriever': 58,
#  'German short haired pointer': 59,
#  'Vizsla': 60,
#  'English setter': 61,
#  'Irish setter': 62,
#  'Gordon setter': 63,
#  'Brittany spaniel': 64,
#  'Clumber': 65,
#  'English springer': 66,
#  'Welsh springer spaniel': 67,
#  'Cocker spaniel': 68,
#  'Sussex spaniel': 69,
#  'Irish water spaniel': 70,
#  'Kuvasz': 71,
#  'Schipperke': 72,
#  'Groenendael': 73,
#  'Malinois': 74,
#  'Briard': 75,
#  'Kelpie': 76,
#  'Komondor': 77,
#  'Old english sheepdog': 78,
#  'Shetland sheepdog': 79,
#  'Collie': 80,
#  'Border collie': 81,
#  'Bouvier des flandres': 82,
#  'Rottweiler': 83,
#  'German shepherd': 84,
#  'Doberman': 85,
#  'Miniature pinscher': 86,
#  'Greater swiss mountain dog': 87,
#  'Bernese mountain dog': 88,
#  'Appenzeller': 89,
#  'Entlebucher': 90,
#  'Boxer': 91,
#  'Bull mastiff': 92,
#  'Tibetan mastiff': 93,
#  'French bulldog': 94,
#  'Great dane': 95,
#  'Saint bernard': 96,
#  'Eskimo dog': 97,
#  'Malamute': 98,
#  'Siberian husky': 99,
#  'Affenpinscher': 100,
#  'Basenji': 101,
#  'Pug': 102,
#  'Leonberg': 103,
#  'Newfoundland': 104,
#  'Great pyrenees': 105,
#  'Samoyed': 106,
#  'Pomeranian': 107,
#  'Chow': 108,
#  'Keeshond': 109,
#  'Brabancon griffon': 110,
#  'Pembroke': 111,
#  'Cardigan': 112,
#  'Toy poodle': 113,
#  'Miniature poodle': 114,
#  'Standard poodle': 115,
#  'Mexican hairless': 116,
#  'Dingo': 117,
#  'Dhole': 118,
#  'African hunting dog': 119}

# It will be easier to shuffle the dataset set if we represent it as a list of filepath/class pairs:
dog_paths_and_classes = []
for dog_class, paths in image_paths.items():
    for path in paths:
        dog_paths_and_classes.append((path, dog_class_ids[dog_class]))

# Next, lets shuffle the dataset and split it into the training set and the test set:
test_ratio = 0.2
train_size = int(len(dog_paths_and_classes) * (1 - test_ratio))
train_size
# Out[27]: 16464

np.random.shuffle(dog_paths_and_classes)
# dog_paths_and_classes[:5]
# Out[30]:
# [('datasets\\dogs\\Images\\n02109047-Great_Dane\\n02109047_2009.jpg', 95),
#  ('datasets\\dogs\\Images\\n02090622-borzoi\\n02090622_8338.jpg', 18),
#  ('datasets\\dogs\\Images\\n02113978-Mexican_hairless\\n02113978_2306.jpg', 116),
#  ('datasets\\dogs\\Images\\n02087046-toy_terrier\\n02087046_4614.jpg', 7),
#  ('datasets\\dogs\\Images\\n02113712-miniature_poodle\\n02113712_2746.jpg', 114)]

dog_paths_and_classes_train = dog_paths_and_classes[:train_size]
dog_paths_and_classes_test = dog_paths_and_classes[train_size:]


#%% preprocess a set of images
# This function will be useful to preprocess the test set, and also to create batches during training.
# For simplicity, we will use the NumPy/SciPy implementation:

from random import sample

def prepare_batch(dog_paths_and_classes, batch_size):
    batch_paths_and_classes = sample(dog_paths_and_classes, batch_size)
    images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch

X_batch, y_batch = prepare_batch(dog_paths_and_classes_train, batch_size=4)

print(X_batch.shape)
print(X_batch.dtype)
print(y_batch.shape)
print(y_batch.dtype)
# (4, 299, 299, 3)
# float32
# (4,)
# int32

# Now let's use this function to prepare the test set:
X_test, y_test = prepare_batch(dog_paths_and_classes_test, batch_size=len(dog_paths_and_classes_test))

X_test.shape
# Out[33]: (4116, 299, 299, 3)

# We could prepare the training set in much the same way, but it would only generate one variant for each image.
# Instead, it's preferable to generate the training batches on the fly during training,
# so that we can really benefit from data augmentation, with many variants of each image.


#%% train the network
#n_epochs = 20
#batch_size = 40
#n_iterations_per_epoch = len(dog_paths_and_classes_train) // batch_size
#INCEPTION_PATH = os.path.join("datasets", "inception")
#INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")
#
#with tf.Session() as sess:
#    init.run()
#    inception_saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
#
#    for epoch in range(n_epochs):
#        print("Epoch", epoch, end="")
#        for iteration in range(n_iterations_per_epoch):
#            print(".", end="")
#            X_batch, y_batch = prepare_batch(dog_paths_and_classes_train, batch_size)
#            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
#
#        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#        print("  Train accuracy:", acc_train)
#
#        save_path = saver.save(sess, "./my_dog_model")
#
#
## accuracy on the test set
#n_test_batches = 30
#X_test_batches = np.array_split(X_test, n_test_batches)
#y_test_batches = np.array_split(y_test, n_test_batches)
#
#with tf.Session() as sess:
#    saver.restore(sess, "./my_dog_model")
#
#    print("Computing final accuracy on the test set (this will take a while)...")
#    acc_test = np.mean([
#        accuracy.eval(feed_dict={X: X_test_batch, y: y_test_batch})
#        for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)])
#    print("Test accuracy:", acc_test)
#
## INFO:tensorflow:Restoring parameters from ./my_dog_model
## Computing final accuracy on the test set (this will take a while)...
## Test accuracy: 0.9018442


#%% classify the images
with tf.Session() as sess:
    saver.restore(sess, "./my_dog_model")
    predictions_val = Y_proba.eval(feed_dict={X: X_test[:10, :, :, :]})

test_image = X_test[0,:,:,:3]
most_likely_class_index = np.argmax(predictions_val[0])
most_likely_class_index
# Out[13]: 79

dog_classes[most_likely_class_index]
# Out[19]: 'Shetland sheepdog'

dog_classes[y_test[0]]
# Out[21]: 'Shetland sheepdog'

plt.imshow(test_image)
plt.title("{}x{}".format(test_image.shape[1], test_image.shape[0]))
plt.axis("off")
plt.show()

top_5 = np.argpartition(predictions_val[0], -5)[-5:]
top_5 = reversed(top_5[np.argsort(predictions_val[0][top_5])])
for i in top_5:
    print("{0}: {1:.2f}%".format(dog_classes[i], 100 * predictions_val[0][i]))

# Shetland sheepdog: 81.38%
# Collie: 18.59%
# Papillon: 0.02%
# Schipperke: 0.00%
# Japanese spaniel: 0.00%



