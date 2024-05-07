# Digit-recognizer

Developed a Digit Recognizer using a **Convolutional Neural Network (CNN)** model. This project aims to accurately classify digits from the dataset, a standard benchmark dataset in the field of machine learning.
<p align="center">
<img src="https://www.lambertleong.com/assets/images/projects/written_digit.gif"/>
</p>

## Model Flow:

### Data Collection and Preprocessing:
* Acquired the dataset from Kaggle, which consists of **28x28 grayscal**e images of handwritten digits (0-9).
* Preprocessed the data by normalizing pixel values to a range between 0 and 1, label encoding and changing  and reshaping images to the appropriate input shape for the CNN model.
### Model Architecture Design:
* Designed a CNN architecture suitable for image classification tasks like digit recognition.
* Typically, the architecture includes **convolutional layers, pooling layers, and fully connected layers**.
### Model Training:
* Split the dataset into training, validation, and test sets.
* Train the CNN model on the training data using an appropriate optimization algorithm like **Adam**.
* Used **categorical_crossentropy** as loss function.
* Monitor the model's performance on the validation set to prevent overfitting.
* Added **Augumentation and dropout** to prevent overfitting.
### Model Evaluation:
* Evaluated the trained model on the test set to assess its generalization performance.
* Used metrics accuracy to quantify performance.
* Visualized training accuracy and validation accuracy, similarly training loss with validation loss
### Model Testing
* For testing purpose, added sample handwritted digits created by me to test.
* Model was able to detect 2 out of 5 images properly.

## Things learnt!
* Basic working of deep learning and CNN basic model
* Normalization ensures that pixel values are within a consistent range, typically between 0 and 1 or -1 and 1.
* Images are typically represented as 2D arrays (height x width x channels). CNNs expect a specific input shape. --> Purpose of reshaping
* CNNs require numerical labels for training.Loss functions (e.g., cross-entropy) compare predicted class probabilities with true labels. --> Purpose of label encoding
## References
https://www.kaggle.com/code/kanncaa1/convolutional-neural-network-cnn-tutorial/notebook
