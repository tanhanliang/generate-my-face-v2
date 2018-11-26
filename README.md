# generate-my-face-v2
This is another attempt at implementing a Boundary Equilibrium Generative Adversarial Network, this time in Pytorch rather than Keras (which was too high level for this). Below are the best images that I managed to create, and some entertaining gifs.

Generated with 180 image dataset:

![alt text](https://github.com/tanhanliang/generate-my-face-v2/raw/master/cool/generated330.png)

Generated with 360 image dataset:

![alt text](https://github.com/tanhanliang/generate-my-face-v2/raw/master/cool/generated-img280.png)

Generated with 3200 image dataset:

![alt text](https://github.com/tanhanliang/generate-my-face-v2/raw/master/cool/generated-img395.png)

![](https://github.com/tanhanliang/generate-my-face-v2/raw/master/cool/3k-dataset-64x64.gif)

Generated with 5900 image dataset:

![alt text](https://github.com/tanhanliang/generate-my-face-v2/raw/master/cool/generated-img219.png)

![](https://github.com/tanhanliang/generate-my-face-v2/raw/master/cool/6k-dataset-128x128.gif)


## How to use this code
Requirements: 
1) Nvidia GPU
2) Installed pytorch with cuda

Steps:
1) Put training set in `data/`
2) Generate more images using functions in `utils.py`, if you want.
3) `python3 training.py`
4) Wait for a very long time. The generated 128x128 image above took about 18 hours of training on a GTX 1080 Ti using a 5900 image dataset.

Note: If you want to decrease the size of the image to say, 64x64 pixels, you have to remove layers in the Autoencoder class, in began.py, to keep the size of the inner representation of the image constant at 8x8.

It is nicer to run the code in `training.py` in a Jupyter Notebook, so that you can visualise the images generated using the `IPython.core.display.display()` function.
