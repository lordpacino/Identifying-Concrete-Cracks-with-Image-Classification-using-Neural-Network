# Identifying Cracks on Concrete with Image Classification using Neural Network


## 1. Summary
The project's goal is to develop a convolutional neural network model capable of detecting cracks in concrete with high accuracy. The issue is represented as a binary classification problem (no cracks/negative and cracks/positive). A dataset of 40000 photos is used to train the model (20000 images of concrete in good condition and 20000 images of concrete with cracks). The source of the dataset is being obtained from https://data.mendeley.com/datasets/5y9wdsg2zt/2 .

## 2. IDE and Framework
Spyder was used as the primary IDE throughout the project. Numpy, Matplotlib, and Tensorflow Keras are the major frameworks utilised in this project.

## 3. Methodology
The methodology for this project is inspired by a documentation on the official TensorFlow website. The documentation can be accesed through here https://www.tensorflow.org/tutorials/images/transfer_learning .

### 3.1. Data Pipeline
The picture data, as well as the labels that go with it, are loaded. The data is initially divided into train-validation sets in a 70:30 ratio. The validation data is then divided into two portions in an 80:20 ratio to obtain test data. The overall split ratio of train-validation-test is 70:24:6. There is no data augmentation because the data quantity and variance are adequate.

### 3.2. Model Pipeline
The input layer is intended to receive coloured pictures with a size of 160x160 pixels. The final form will be (160,160,3).

This project's deep learning model is built via transfer learning. To begin, a preprocessing layer is established, which changes the pixel values of the input pictures to a range of -1 to 1. This layer functions as a feature scaler and is required for the transfer learning model to produce the right signals.

A pretrained MobileNet v2 model is employed for feature extraction. With ImageNet pretrained parameters, the model is readily available within the TensorFlow Keras package. It's also locked, so it won't update throughout model training.

As the classifier, a global average pooling and dense layer are employed to generate softmax signals. The predicted class is identified using the softmax signals.

The model is shown in figure in a simplified form in the image below.

[image](https://user-images.githubusercontent.com/76200485/165531825-a282c5ee-9f51-4ec1-afd3-cb1ac3e91a0c.png)

The model is trained with a batch size of 16 and 10 epochs. After training, the model reaches 99% training accuracy and 95% validation accuracy. The training results are shown in the figures below.

[epochloss](https://user-images.githubusercontent.com/76200485/165533669-e4a760a0-cfb7-4003-9ad1-195de6cd1e7a.jpg)
![epochaccracy](https://user-images.githubusercontent.com/76200485/165533798-dbcdd6bc-b3cb-45d7-a4a0-93d0ca5bfd4c.jpg)

The graphs clearly show evidence of convergence, showing that the model has been trained to achieve an ideal level of accuracy and loss.

## 4. Results
With the test data, the model is assessed. The loss and accuracy are displayed in the image below.

![result](https://user-images.githubusercontent.com/76200485/165534494-ef3cdd78-42c5-4976-b6bd-f6dc125fc2fc.jpg)

The model is also used to make predictions, which are then compared to the actual findings.

![image](https://user-images.githubusercontent.com/76200485/165533992-e7a06fdc-f490-4083-ae88-69945144463b.png)
