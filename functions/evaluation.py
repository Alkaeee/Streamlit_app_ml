import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras

def evaluate_model(model, val_ds):
    """
    Evaluate a machine learning model on a validation dataset.

    Parameters:
    - model (object): The model to evaluate.
    - val_ds (tf.data.Dataset): Validation dataset containing pairs of images and labels.

    Returns:
    - y_pred (array): Model predictions on the validation dataset.
    - ev_model (float): Model evaluation metric on the validation dataset.
    """
    X_test, y_test = [], []

    for images, labels in val_ds:
        X_test.append(images.numpy())
        y_test.append(labels.numpy()) 

    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    y_pred = model.predict(X_test)
    ev_model = model.evaluate(X_test, y_test)

    return y_pred, ev_model



def evaluate_img(image, model, class_names):
    """
    Evaluates an image using a trained Keras model and displays the image with predictions.

    Parameters:
    - img: The input image to be evaluated.
    - model (tf.keras.Model): The trained Keras model.
    - class_names (list): A list of class names corresponding to the model's output classes.

    Returns:
    - np.ndarray: Predictions for each class.
    - str: Predicted label name.

    Example:
    ```python
    predictions, predicted_label_name = evaluate_img(input_image, trained_model, class_names)
    ```

    The `evaluate_img` function performs the following operations:
    1. Resizes the input image to the required size (250x250).
    2. Displays the resized image using Matplotlib.
    3. Converts the image to a NumPy array and adds an extra dimension.
    4. Uses the trained model to predict the class probabilities.
    5. Identifies the predicted label and its corresponding class name.

    Note:
    - Assumes that TensorFlow (`tf`), Keras (`keras`), Matplotlib (`plt`), and NumPy (`np`) have been installed.

    """
    image.save(f"test.{image.format}")

    img = keras.utils.load_img(
        f"test.{image.format}", target_size=(250,250)
    )
    img = img.resize((250,250))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  

    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions)
    predicted_label_name = class_names[predicted_label]

    return predictions, predicted_label_name