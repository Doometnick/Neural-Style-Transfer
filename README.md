# Neural-Style-Transfer
[Neural Style Transfer (NST)](https://arxiv.org/abs/1508.06576) extracts an art style from one picture (style picture) and applies it to another picture (content picture).  


***
![](https://github.com/Doometnick/Neural-Style-Transfer/blob/master/samples/smpl5.png)
![](https://github.com/Doometnick/Neural-Style-Transfer/blob/master/samples/smpl1.png)
![](https://github.com/Doometnick/Neural-Style-Transfer/blob/master/samples/smpl2.png)
![](https://github.com/Doometnick/Neural-Style-Transfer/blob/master/samples/smpl3.png)
![](https://github.com/Doometnick/Neural-Style-Transfer/blob/master/samples/smpl4.png)  
Image sources: [pexels](https://www.pexels.com/), [Google](https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg) 
***

This repository allows to run NST and apply several settings without having to understand the machine learning techniques behind or digging into the code. 

The technique is inspired by [Gatys, Egger, Bethge](https://arxiv.org/abs/1508.06576) with some code samples taken from [Google's Tensorflow Examples](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/style_transfer.ipynb). In addition, packaging and several launch parameters have been added for an easy use. Especially the layer configuration in _config.json_ allows to deviate from the original paper and create very unique styles.

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

It is heavily recommended to run the example with an enabled GPU, since training on the CPU is significantly slower.
