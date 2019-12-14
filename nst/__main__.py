import argparse

parser = argparse.ArgumentParser(
    description="Apply artistic styles from one picture to another.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Required arguments
parser.add_argument("content_image", type=str,
                    help="Which image in folder 'img' will be used for the content.")
parser.add_argument("style_image", type=str, 
                    help="Which image in folder 'img' will be used for the style.")

# Optional arguments
parser.add_argument("--epochs", type=int, default=10, 
                    help="How many epochs will be trained.")
parser.add_argument("--export_after_each_epoch", type=bool, default=True, 
                    help="Export a stylized image after every epoch.")
parser.add_argument("--learning_rate", type=float, default=0.02, 
                    help="Learning rate of optimizer.")
parser.add_argument("--max_img_dim", type=float, default=512.0, 
                    help="Resize the image's longest dimension in pixels."
                    "The image aspect ratio will be kept. Note that this value should not "
                    "be larger than the maximum of the original's image dimensions.")
parser.add_argument("--weight_content_loss", type=float, default=1e4,
                    help="Contribution of content loss to total loss while training.")
parser.add_argument("--weight_style_loss", type=float, default=0.1,
                    help="Contribution of style loss to total loss while training.")
parser.add_argument("--weight_total_variation_loss", type=float, default=30,
                    help="Contribution of total variation loss to total loss while training.")
parser.add_argument("--config", type=str, default='default',
                    help="Specifies which layers are used for style and content extraction. "
                    "Can be found in 'config.json'.")
args = parser.parse_args()

from model import Model
import images as images

style_img = images.load_img(args.style_image, args.max_img_dim)
content_img = images.load_img(args.content_image, args.max_img_dim)

model = Model(content_image=content_img,
              style_image=style_img,
              learning_rate=args.learning_rate,
              style_weight=args.weight_style_loss,
              content_weight=args.weight_content_loss,
              total_variation_weight=args.weight_total_variation_loss,
              layer_config=args.config)
model.train(epochs=args.epochs)
