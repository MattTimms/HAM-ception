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

## Results
The following is in reference to the figure shown below. Session0 contained a pre-trained InceptionV3 model with
 a new trainable fully-connected final layer. Results after 10 epochs were above benchmark accuracy.
 Session1 increase training iterations to 25 epochs; accuracy saturated. Session2 unfroze model layers proceeding 
 Conv2d_4a_3x3 and gave best results - No further training has since been conducted.  
 Tests are conducted on 32 samples not seen during training.
 
```
Session2 Results
Training:
    Loss: 0.0017 Acc: 0.9807
Testing:
    Loss: 0.0053 Acc: 0.9375
```   

![Model results image... should've loaded here](https://github.com/MattTimms/HAM10000/blob/master/images/plot.png)

## Considerations
The HAM dataset is heavily skewed (see figure below), *Melanocytic Nevi* accounts for ~67% of the dataset. 
 Therefore, it is reasonable to set this percentage as the accuracy benchmark for training; a trained network
 must have greater accuracy then if one was to trivially predict all samples as samples of *Melanocytic Nevi*.  
![HAM Data Distribution image... should've loaded here](https://github.com/MattTimms/HAM10000/blob/master/images/data_distribution.png)

## TensorBoard
I had an issue hosting TensorBoard on my local machine; this command is a work-around.  
````tensorboard --logdir=. --host localhost --port 6006````