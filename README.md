# Facial Emotion Recognition

## Introduction

This project is a facial emotion detection system that uses a convolutional neural network to detect facial emotions. 
The model is trained on the [FER-2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) which consists of 48x48 pixel grayscale images of faces with 7 different emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The model is trained using PyTorch and the dataset is preprocessed using torchvision image transforms.

## Web Application

The web application backend is built using Flask and the front-end is created using SvelteJS. 
The application allows the user to either capture an image of themselves or upload an image and the model will predict the emotion of the person in the image.

The application is live and can be accessed at [https://fer.rohand.in](https://fer.rohand.in/).

## Usage

To run the models, clone the repository and run the following command in the terminal:

```shell
python -m pip install -r requirements.txt
```

Then, run the following command to make a prediction on an image:

```shell
python predict.py --image-file <path-to-image> --model-path model/SimpleCNNMdodel
```

## Models

### SimpleCNNModel

This model is a simple convolutional neural network with 3 convolutional layers with ReLU activation, MaxPooling and BatchNormalization and 4 fully connected layers. 
The model is trained for 3 epochs with a batch size of 32 and a learning rate of 0.001 for the Adam optimizer. 
The model achieves a validation accuracy of 56%.

## References
- Deploy multiple Flask applications in the same server: [here](https://towardsdatascience.com/deploy-multiple-flask-applications-using-nginx-and-gunicorn-16f8f7865497)
- SvelteJS + Flask: [here](https://cabreraalex.medium.com/svelte-js-flask-combining-svelte-with-a-simple-backend-server-d1bc46190ab9)
