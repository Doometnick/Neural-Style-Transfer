""" This is a self-written version of Google's tutorial about
Neural Style Transfer which I made to gain a better understanding
of the code and the content.

It extracts style features from one picture and applies it to another
picture while keeping the other picture's content unchanged.

The original code can be found here: 
https://www.tensorflow.org/tutorials/generative/style_transfer
"""
def run():
    print("Called as main")
    import argparse
    from nst.model import Model
    import nst.images as images

    style_path = "style_image.jpg"
    content_path = "eyes_female.jpg"

    #content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    #style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

    style_img = images.load_img(style_path)
    content_img = images.load_img(content_path)

    model = Model(content_image=content_img,
                  style_image=style_img)
    model.train(epochs=5)
