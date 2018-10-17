# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 21:56:39 2018

@author: Rangel
"""

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

img_str = "test_image1.jpg"
img_path = os.path.join("datasets", "dogs")
path = os.path.join(img_path, img_str)
dog_classes = ['Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih tzu', 'Blenheim spaniel', 'Papillon', 'Toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'Basset', 'Beagle', 'Bloodhound', 'Bluetick', 'Black and tan coonhound', 'Walker hound', 'English foxhound', 'Redbone', 'Borzoi', 'Irish wolfhound', 'Italian greyhound', 'Whippet', 'Ibizan hound', 'Norwegian elkhound', 'Otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'American staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'Wire haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Airedale', 'Cairn', 'Australian terrier', 'Dandie dinmont', 'Boston bull', 'Miniature schnauzer', 'Giant schnauzer', 'Standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'Silky terrier', 'Soft coated wheaten terrier', 'West highland white terrier', 'Lhasa', 'Flat coated retriever', 'Curly coated retriever', 'Golden retriever', 'Labrador retriever', 'Chesapeake bay retriever', 'German short haired pointer', 'Vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'Clumber', 'English springer', 'Welsh springer spaniel', 'Cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'Kuvasz', 'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Kelpie', 'Komondor', 'Old english sheepdog', 'Shetland sheepdog', 'Collie', 'Border collie', 'Bouvier des flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'Miniature pinscher', 'Greater swiss mountain dog', 'Bernese mountain dog', 'Appenzeller', 'Entlebucher', 'Boxer', 'Bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great dane', 'Saint bernard', 'Eskimo dog', 'Malamute', 'Siberian husky', 'Affenpinscher', 'Basenji', 'Pug', 'Leonberg', 'Newfoundland', 'Great pyrenees', 'Samoyed', 'Pomeranian', 'Chow', 'Keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'Toy poodle', 'Miniature poodle', 'Standard poodle', 'Mexican hairless', 'Dingo', 'Dhole', 'African hunting dog']
logdir = os.path.join(img_path, "tf_logs")

from scipy.misc import imresize
from skimage.transform import resize
import matplotlib.image as mpimg

def prepare_image(image, target_width=299, target_height=299):
    # Now, let's resize the image to the target dimensions.
    image = imresize(image, (target_width, target_height))

    # Finally, let's ensure that the colors are represented as
    # 32-bit floats ranging from 0.0 to 1.0 (for now):
    return image.astype(np.float32) / 255


def prepare_batch(dog_paths=path, batch_size=1):
    images = [mpimg.imread(dog_paths)[:, :, :3]]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1

    # visualize the test image
    plt.imshow(images[0])
    plt.title("{}x{}".format(images[0].shape[1], images[0].shape[0]))
    plt.axis("off")
    plt.show()

    return X_batch

def predict_dog_breed(path):
    X_test = prepare_batch(path)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("./my_dog_model.meta")
        saver.restore(sess, "./my_dog_model")
        # file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        X = tf.get_default_graph().get_tensor_by_name("X:0")
        Y_proba = tf.get_default_graph().get_tensor_by_name("new_output_layer/Y_proba:0")
        predictions_val = Y_proba.eval(feed_dict={X: X_test})

    most_likely_class_index = np.argmax(predictions_val[0])
    print(dog_classes[most_likely_class_index])
    return predictions_val

def most_likely(predict):
    top_5 = np.argpartition(predict[0], -5)[-5:]
    top_5 = reversed(top_5[np.argsort(predict[0][top_5])])
    for i in top_5:
        print("{0}: {1:.2f}%".format(dog_classes[i], 100 * predict[0][i]))

