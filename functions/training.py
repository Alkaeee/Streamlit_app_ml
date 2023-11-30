import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.python.client import device_lib
import variables as v


def use_gpu():
    """
    Configures TensorFlow to use GPU(s) and prints information about available GPUs.

    Example:
    ```python
    use_gpu()
    ```

    The `use_gpu` function performs the following operations:
    1. Lists and sets memory growth for available physical GPUs.
    2. Prints the number of available GPUs and their descriptions.
    3. Lists logical GPUs and prints the number of physical and logical GPUs.

    Note:
    - Assumes that TensorFlow (`tf`) has been installed.
    - Assumes that the GPU is properly configured on the system.

    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('  GPU: {}'.format([x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']))

    gpus = tf.config.list_physical_devices('GPU')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpus_logicas = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "GPUs físicas,", len(gpus_logicas), "GPUs lógicas")
        except RuntimeError as e:
            print(e)

def create_train_val(image_size = (250,250), batch_size = 16):
    """
    Creates TensorFlow datasets for training and validation using images from a directory.

    Parameters:
    - image_size (tuple): The desired image size in the format (height, width).
    - batch_size (int): Batch size for training and validation datasets.

    Returns:
    - tf.data.Dataset: Training dataset.
    - tf.data.Dataset: Validation dataset.

    Example:
    ```python
    train_dataset, val_dataset = create_train_val(image_size=(250, 250), batch_size=16)
    ```

    The `create_train_val` function performs the following operations:
    1. Reads images from the specified directory using `tf.keras.utils.image_dataset_from_directory`.
    2. Splits the dataset into training and validation sets using a validation split of 20%.
    3. Applies data augmentation techniques (random flip, rotation, zoom) to the training dataset.
    4. Prefetches samples in GPU memory to maximize GPU utilization.

    Note:
    - Assumes that images are organized in subdirectories representing different classes.
    - Assumes that TensorFlow (`tf`) and Keras (`keras`) have been installed.

    """
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "..\data\\raw_data\\image_resize",
        label_mode='categorical',
        validation_split=0.2,
        subset="both",
        seed=666,
        image_size=image_size,
        batch_size=batch_size,
    )

    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                        input_shape=(250,
                                    250,
                                    3)),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ]
    )

    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

def make_model(input_shape, num_classes):
    """
    Creates a custom convolutional neural network (CNN) model using Keras.

    Parameters:
    - input_shape (tuple): The input shape of the model in the format (height, width, channels).
    - num_classes (int): The number of output classes.

    Returns:
    - tf.keras.Model: The created CNN model.

    Example:
    ```python
    model = make_model(input_shape=(250, 250, 3), num_classes=10)
    ```

    The `make_model` function defines a custom CNN model with the following architecture:
    1. Entry block with rescaling, convolution, batch normalization, and activation.
    2. Multiple blocks with separable convolutions, batch normalization, and residual connections.
    3. Global average pooling layer.
    4. Dense layers with dropout for classification.

    Note:
    - Assumes that TensorFlow (`tf`) and Keras (`keras`) have been installed.

    """    
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [250, 250, 3]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    
    activation = "softmax"
    units = num_classes

    x = layers.Dropout(0.3)(x)
    x = keras.layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(500, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

def trained_model(model, train_ds, val_ds, folder_name, epochs=25, patience=20):
    """
    Trains a Keras model on a training dataset and evaluates it on a validation dataset.

    Parameters:
    - model (tf.keras.Model): The Keras model to be trained.
    - train_ds (tf.data.Dataset): The training dataset.
    - val_ds (tf.data.Dataset): The validation dataset.
    - folder_name (str): The folder name to save the trained model.
    - epochs (int): Number of epochs for training (default is 25).
    - patience (int): Number of epochs for early stopping patience (default is 20).

    Returns:
    - tf.keras.Model: The trained Keras model.
    - History: The training history.

    Example:
    ```python
    trained_model, history = trained_model(model, train_dataset, val_dataset, 'my_model', epochs=25, patience=20)
    ```

    The `trained_model` function performs the following operations:
    1. Configures the model for training using the Adam optimizer and categorical crossentropy loss.
    2. Defines callbacks for model checkpointing and early stopping.
    3. Fits the model to the training dataset, with validation on the validation dataset.
    4. Returns the trained model and the training history.

    Note:
    - Assumes that TensorFlow (`tf`) and Keras (`keras`) have been installed.
    - Assumes that the constants module or file (`v`) is correctly imported.

    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{v.MODELS_PATH}\\{folder_name}"),
        keras.callbacks.EarlyStopping(patience)
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    return model, history

def plot_history_loss(history):
    """
    Plots the training and validation loss and accuracy over epochs.

    Parameters:
    - history (tf.keras.callbacks.History): The training history obtained from model training.

    Example:
    ```python
    plot_history_loss(history)
    ```

    The `plot_history_loss` function performs the following operations:
    1. Extracts training and validation loss and accuracy from the training history.
    2. Identifies the epoch with the lowest validation loss and the epoch with the highest validation accuracy.
    3. Plots two subplots: one for loss and another for accuracy, with training and validation curves.
    4. Highlights the points for the epochs with the lowest validation loss and highest validation accuracy.

    Note:
    - Assumes that Matplotlib (`plt`) and NumPy (`np`) have been installed.
    - Assumes that the training history contains 'accuracy', 'loss', 'val_accuracy', and 'val_loss'.

    """
    tr_acc = history.history['accuracy']
    tr_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    # Plot training history
    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')
    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    plt.show()

