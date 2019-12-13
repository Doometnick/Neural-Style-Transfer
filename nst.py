""" This is a self-written version of Google's tutorial about
Neural Style Transfer which I made to gain a better understanding
of the code and the content.

It extracts style features from one picture and applies it to another
picture while keeping the other picture's content unchanged.

The original code can be found here: 
https://www.tensorflow.org/tutorials/generative/style_transfer
"""


import os

import matplotlib.pyplot as plt
import PIL.Image
import tensorflow as tf
import numpy as np
import IPython.display as display

style_path = "style_image.jpg"
content_path = "Owl.jpg"

#content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

#style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

def load_img(img_path):
    max_dim = 512
    img = tf.io.read_file(img_path)
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

style_img = load_img(style_path)
content_img = load_img(content_path)

# Define the layers that we want to extract:
style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
content_layers = ["block5_conv2"]

def get_vgg_model(layers):
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layers]
    return tf.keras.Model([vgg.input], outputs)

# We need to describe content and style features from the outputs of the 
# vgg layers. Content features can be described simply by the feature maps, ie
# by the output of the content layers.
# Style features can be described by calculating means and correlations
# across different features maps.

def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor) 
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()

        self.vgg = get_vgg_model(style_layers + content_layers)

        self.style_layers = style_layers
        self.content_layers = content_layers   

        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)     

    def __call__(self, x):
        x = x * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(x)

        output = self.vgg(preprocessed_input)
        style_outputs = output[:self.num_style_layers]
        content_outputs = output[self.num_style_layers:]
        
        content_results = {
            name: values
            for name, values
            in zip(content_layers, content_outputs)
        }

        style_results = {
            name: gram_matrix(values)
            for name, values
            in zip(style_layers, style_outputs)
        }
        return {"content": content_results,
                "style": style_results}


extractor = StyleContentModel(style_layers, content_layers)

# Benchmarks for style and content
style_target = extractor(style_img)["style"]
content_target = extractor(content_img)["content"]

w_content, w_style = 1e4, 1e-2

def style_content_loss(content_outputs, style_outputs):
    # Calculate individual losses per layer and sum them up.
    content_loss = tf.add_n(
        [tf.reduce_mean((content_outputs[name] - content_target[name]) ** 2)
         for name in content_outputs.keys()]
    )
        
    style_loss = tf.add_n(
        [tf.reduce_mean((style_outputs[name] - style_target[name]) ** 2)
         for name in style_outputs.keys()]
    )
    
    content_loss *= w_content / len(content_layers)
    style_loss *= w_style / len(style_layers)

    return content_loss + style_loss

def clip_0_1(img):
    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)

sample_output = extractor(content_img)
style_content_loss(sample_output["content"], sample_output["style"])

image = tf.Variable(content_img)

optimizer = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.99, epsilon=1e-1)

@tf.function()
def training_step(img):
    with tf.GradientTape() as tape:
        outputs = extractor(img)
        loss = style_content_loss(outputs["content"],
                                  outputs["style"])
        loss += 30 * tf.image.total_variation(img)
    
    gradients = tape.gradient(loss, img)
    optimizer.apply_gradients([(gradients, img)])
    img.assign(clip_0_1(img))

epochs = 10
steps_per_epoch = 100
step = 0
for epoch in range(epochs):
    for _ in range(steps_per_epoch):
        step += 1
        training_step(image)
    #display.clear_output(wait=True)
    #display.display(tensor_to_image(image))
    print(f"Step {step}")
    tensor_to_image(image).save(f"stylized_image_{epoch + 1}.jpg")
