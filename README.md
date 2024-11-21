# Deep Learning based American Sign Language Recognition

Final Deep Learning Project for USC EE-541 (A Computational Introduction to Deep Learning) _**✨Received extra credit for this project🚀**_
#### [Medium blog post](https://medium.com/@sudarshanasrao/bridging-communication-gaps-using-deep-learning-for-american-sign-language-recognition-34bbd089f465)

## Overview

### Background

Sign language is a prominent form of communication for deaf communities across the world. However, there is a limited number of people outside of these communities who understand sign language. Therefore, there exists a need for a translation mechanism to convert sign language into a form that can readily understood by the general public. Furthermore, the nature of the sign language used is dependent upon region and, as such, no universal sign language exists. In the United States, American Sign Language (ASL) is the most widely used form and generally uses individual hand signs to convey full words. Fingerspelling is used in cases for which a sign for a word does not exist by signing individual letters.

### Objective

This project aims to develop a Deep Learning model to translate sign language to a text-based representation where the scope of the input is limited to the ASL alphabet used in fingerspelling. Specifically, when given a single image containing a hand sign, the model should classify the image according to the ASL alphabet with a high degree of accuracy. In particular, this project aims to investigate various Deep Learning model approaches in order to select the best model for this task and evaluate its ability to generalize to unseen ASL data.

### Dataset

The dataset (training & testing combined) is 1.03 GB. It consists of 87,209 images (training & testing combined) in jpeg format. Each image is in RGB (Red-Green-Blue) scale and its dimensions are 200x200 pixels. The dataset folder has two sub-directories, namely the training dataset and the testing dataset. The training dataset contains one sub-folder for each of the 29 classes. The 29 classes are the 26 English alphabet characters (’a’ through ’z’), space, delete, and nothing. There are 3000 training images for each class. The testing dataset consists of 29 images (i.e., one image for each class). Two supplementary real world datasets were also captured.

The Kaggle dataset can be accessed [here](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). The supplementary dataset is available on request.

## Getting Started

### Running a model

1. Run the *init.ipynb* script to download the Kaggle dataset. A username and token is required and can be generated from your user profile on Kaggle.
2. Run any of the *model_{name}.ipynb* scripts in the root directory to load, train and evaluate that model. All these scripts have the same structure with the model hyperparameters declared as constants at the top of the scripts. Each script corresponds to a different model. Note, the dataset sources should be set to access the datasets available on the machine where the script executes.

### Structure

Only the model and initialization scripts are directly in the root directory. All other methods and classes required by the model scripts are stores in the *utils* package. The *utils* package contains the following modules:

- *data.py* - Methods for manipulating, saving and visualizing the dataset.
- *eval.py* - Methods for evaluating a model and visualizing the evaluation metrics.
- *models.py* - Model class definitions. 
- *train.py* - Methods for training a model on a given dataset.
