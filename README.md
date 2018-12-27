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

## Considerations
The HAM dataset is heavily skewed; see Fig. 1. *Melanocytic Nevi* accounts for ~66% of the dataset. 
Therefore, it is reasonable to set this as the accuracy benchmark for training; a trained network must
greater accuracy then to trivally predict all samples as exhibiting *Melanocytic Nevi*.  
![HAM Data Distribution image... should've loaded here](https://github.com/MattTimms/HAM10000/blob/master/images/data_distribution.png)

## TensorBoard
I had an issue hosting TensorBoard on my local machine; this command is a work-around.  
````tensorboard --logdir=. --host localhost --port 6006````