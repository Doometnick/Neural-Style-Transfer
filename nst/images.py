from os import path, makedirs

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf


def load_img(img_path, max_dim):
    img = tf.io.read_file(path.join("img", img_path))
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)

    shape = tf.shape(img)[:-1]
    shape = tf.cast(shape, tf.float32)
    longest_dim = max(shape)
    scale = max_dim / longest_dim

    new_shape = shape * scale
    new_shape = tf.cast(new_shape, tf.int32)

    img = tf.image.resize(img, new_shape)
    # Add first axis as required by CNNs
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) == 4:
        if tensor.shape[0] != 1:
            raise ValueError("First dimension different from what's expected (1).")
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def plot_img_from_tensor(tensor):
    if len(tensor.shape) == 4:
        tensor = tf.squeeze(tensor, axis=0)
    plt.imshow(tensor)
    plt.show()

def save_image(tensor, img_name):
    export_folder = "img_stylized"
    if not path.exists(export_folder):
        makedirs(export_folder)
    tensor_to_image(tensor).save(path.join(export_folder, img_name))
