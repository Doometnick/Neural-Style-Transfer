# Neural-Style-Transfer
[Neural Style Transfer (NST)](https://arxiv.org/abs/1508.06576) extracts an art style from one picture (style picture) and applies it to another picture (content picture). 

This repository allows to run NST and apply several settings without having to understand the machine learning techniques behind or digging into the code. 

The technique is inspired by [Gatys, Egger, Bethge](https://arxiv.org/abs/1508.06576) with some code samples taken from [Google's Tensorflow Examples](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/style_transfer.ipynb). In addition, packaging and several launch parameters have been added for an easy use.

# How To Run
_The repository has only been tested on Windows._

Make sure requirements are satisfied by running  
`pip install -r requirements.txt`.

Then simply run the following in the command line:
`python nst <content_image> <style_image>`.  

For example:  
`python nst img1.jpg style_image.jpg`  

Both images have to be located in the *.img/* folder, the resulting images will be saved in the *img_stylized* folder.  
See `python nst --help` for further details and optional parameters.
