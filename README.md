# *HAM-ception*

````
InceptionV3 w/ a side of HAM
Written by Matthew Timms for DeepNeuron-AI  

Image classification of HAM10000 dataset using pre-trained InceptionV3.  

Usage:
    main.py [options]  
    main.py (-h | --help)  
     
Example:  
    python main.py --cuda --training --dataroot ./dataset/skin-cancer-mnist-ham10000/ --workers=8  
````

## Setup
````pip install -r requirements.txt````  
Download the HAM10000 dataset from [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/home).
Unzip and set --model_path to its directory.

## TensorBoard
I had an issue hosting TensorBoard on my local machine; this command is a work-around.  
````tensorboard --logdir=. --host localhost --port 6006````