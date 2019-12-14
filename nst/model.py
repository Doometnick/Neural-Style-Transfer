import json
import tensorflow as tf
from images import save_image

if not tf.test.is_gpu_available():
    print("It is recommended to use a GPU to speed up the training.")


class StyleContentModel(tf.keras.models.Model):
    """https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/style_transfer.ipynb"""
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()

        self.vgg = self._get_vgg_model(style_layers + content_layers)
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
            in zip(self.content_layers, content_outputs)
        }

        style_results = {
            name: self._gram_matrix(values)
            for name, values
            in zip(self.style_layers, style_outputs)
        }
        return {"content": content_results,
                "style": style_results}


    def _get_vgg_model(self, layers):
        vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layers]
        return tf.keras.Model([vgg.input], outputs)
   
    def _gram_matrix(self, input_tensor):
        result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor) 
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations


class Model:

    def __init__(self, 
                 content_image, 
                 style_image, 
                 content_weight=1e4,
                 style_weight=1e-2,
                 total_variation_weight=30,
                 learning_rate=0.01, 
                 layer_config="default"):
        self.content_img = content_image
        self.style_image = style_image

        try:
            layer_cfg= json.load(open("nst/config.json", "r"))
            style_layers = layer_cfg[layer_config]["style_layers"]
            content_layers = layer_cfg[layer_config]["content_layers"]
        except KeyError:
            print("Incorrect config names, check for typos.")
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight

        self.n_content_layers = len(content_layers)
        self.n_style_layers = len(style_layers)

        self.extractor = StyleContentModel(style_layers, content_layers)
        self.style_target = self.extractor(style_image)["style"]
        self.content_target = self.extractor(content_image)["content"]

        # Variable used for training
        self.image = tf.Variable(content_image)

        self.optimizer = self.set_optimizer()

    def set_optimizer(self, learning_rate=0.01):
        return tf.optimizers.Adam(learning_rate, beta_1=0.99, epsilon=0.1)
    
    @tf.function()
    def training_step(self, img):
        with tf.GradientTape() as tape:
            outputs = self.extractor(img)
            loss = self._style_content_loss(outputs["content"],
                                      outputs["style"])
            loss += self.total_variation_weight * tf.image.total_variation(img)

        gradients = tape.gradient(loss, img)
        self.optimizer.apply_gradients([(gradients, img)])
        img.assign(self._clip_0_1(img))

    def train(self, epochs=10, steps_per_epoch=100, export_every_epoch=True):
        step = 0
        for epoch in range(epochs):
            for _ in range(steps_per_epoch):
                step += 1
                self.training_step(self.image)
            print(f"Step {step}")
            if export_every_epoch is True:
                save_image(self.image, f"img_{epoch + 1}.jpg")
    def _style_content_loss(self, content_outputs, style_outputs):
        # Calculate individual losses per layer and sum them up.
        content_loss = tf.add_n(
            [tf.reduce_mean((content_outputs[name] - self.content_target[name]) ** 2)
             for name in content_outputs.keys()]
        )

        style_loss = tf.add_n(
            [tf.reduce_mean((style_outputs[name] - self.style_target[name]) ** 2)
             for name in style_outputs.keys()]
        )

        content_loss *= self.content_weight / self.n_content_layers
        style_loss *= self.style_weight / self.n_style_layers

        return content_loss + style_loss

    def _clip_0_1(self, img):
        return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)
