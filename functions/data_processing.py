import pandas as pd
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import shutil

import variables as v



def resize_data(path):
    """
    Resize images in a directory structure and save the resized versions in another directory.

    Parameters:
    - path (str): Path to the main directory containing subdirectories with images.

    Example:
    ```python
    resize_data('/path/to/main/directory')
    ```

    The `resize_data` function performs the following operations:
    1. Iterates over subdirectories in the given directory (`path`).
    2. For each subdirectory, iterates over image files.
    3. Attempts to load each image, resize it to 250x250 pixels, and save it with a new name.
    4. Moves the resized image to the destination directory (`v.RESIZE_IMAGE_PATH`), creating directories if necessary.
    5. Prints an error message in case of an exception and continues with other images.

    Raises:
    - Exception: Can occur if there are issues loading or processing the images.

    Note:
    - Make sure to have the necessary libraries installed, such as `opencv` and `shutil`.
    - Ensure that you have correctly defined the destination path `v.RESIZE_IMAGE_PATH` and import your constants.

    """
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            try:
                file_path = os.path.join(folder_path, file)
                img = cv2.imread(file_path)
                img = cv2.resize(img, (250, 250))
                name_arch = file.split(".")
                cv2.imwrite(f"{name_arch[0]}_resize.png", img)
                destination_path = os.path.join(v.RESIZE_IMAGE_PATH, folder)
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)
                    print("Create dir: ", destination_path)
                shutil.move(f"{name_arch[0]}_resize.png", destination_path)
            except Exception as e:
                print("Error: ",e)

def compressed_data(path):
    """
    Compresses images in a directory structure using Principal Component Analysis (PCA) 
    and saves the compressed versions in another directory.

    Parameters:
    - path (str): Path to the main directory containing subdirectories with images.

    Example:
    ```python
    compressed_data('/path/to/main/directory')
    ```

    The `compressed_data` function performs the following operations:
    1. Iterates over subdirectories in the given directory (`path`).
    2. For each subdirectory, iterates over image files.
    3. Loads each image, resizes it to 50x50 pixels.
    4. Applies PCA separately to the red, green, and blue channels.
    5. Inverts the PCA transformation to reconstruct the channels.
    6. Merges the inverted channels to create the compressed image.
    7. Saves the compressed image with a new name.
    8. Moves the compressed image to the destination directory (`v.COMPRESSED_IMAGE_PATH`), creating directories if necessary.

    Raises:
    - Exception: Can occur if there are issues loading or processing the images.

    Note:
    - Make sure to have the necessary libraries installed, such as `opencv`, `numpy`, and `scikit-learn`.
    - Ensure that you have correctly defined the destination path `v.COMPRESSED_IMAGE_PATH` and import your constants.

    """
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        print(folder_path)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (50, 50))

            blue, green, red = cv2.split(img)
            # Initialize PCA with first 50 principal components
            pca = PCA(50)
            
            #Applying to red channel and then applying inverse transform to transformed array.
            red_transformed = pca.fit_transform(red)
            red_inverted = pca.inverse_transform(red_transformed)

            #Applying to Green channel and then applying inverse transform to transformed array.
            green_transformed = pca.fit_transform(green)
            green_inverted = pca.inverse_transform(green_transformed)
            
            #Applying to Blue channel and then applying inverse transform to transformed array.
            blue_transformed = pca.fit_transform(blue)
            blue_inverted = pca.inverse_transform(blue_transformed)

            img_compressed = (cv2.merge([red_inverted, green_inverted, blue_inverted])).astype(np.uint8)
            destination_path = os.path.join(v.COMPRESSED_IMAGE_PATH, folder)

            name_arch = file.split(".")
            cv2.imwrite(f"{name_arch[0]}_compressed.png", img_compressed)

            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            shutil.move(f"{name_arch[0]}_compressed.png", destination_path)



def create_dataframe(path):
    """
    Creates a pandas DataFrame from images in a directory structure, 
    flattening the pixel values and including the corresponding image names.

    Parameters:
    - path (str): Path to the main directory containing subdirectories with images.

    Returns:
    - pd.DataFrame: A DataFrame containing flattened pixel values and image names.

    Example:
    ```python
    dataframe = create_dataframe('/path/to/main/directory')
    ```

    The `create_dataframe` function performs the following operations:
    1. Iterates over subdirectories in the given directory (`path`).
    2. For each subdirectory, iterates over image files.
    3. Loads each image and flattens its pixel values.
    4. Checks if the flattened pixel values have a length of 7500.
    5. If the condition is met, adds the flattened pixel values and the image name to the DataFrame.
    6. Randomly shuffles the rows of the DataFrame.
    
    Note:
    - Assumes that the images have a fixed size of 50x50 pixels with 3 color channels.

    Returns a DataFrame with columns representing flattened pixel values (from 0 to 7499) 
    and a column 'name_img' containing the corresponding image names.

    """
    print(path)
    dict_comp_flat_px = []
    names = []
    for name_rut in os.listdir(path):
        folder_path = os.path.join(path, name_rut)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            img = cv2.imread(file_path)
            pix_val =np.array(img)
            name_arch = file.split(".")
            name_arch = name_arch[0].split("_")
            name_arch = name_arch[0]
            if len(pix_val.flatten()) == 7500 or pix_val.size == 7500:
                dict_comp_flat_px.append(pix_val.flatten())
                names.append(name_rut)
            else:
                print(pix_val.size)
                print(name_arch)
    df_comp = pd.DataFrame(dict_comp_flat_px, columns=np.arange(0,7500))
    df_comp["name_img"] = names

    df_comp = df_comp.sample(frac=1).reset_index(drop=True)

    return df_comp

def feauture_engine(df):
    """
    Encodes the 'name_img' column in a pandas DataFrame using Label Encoding.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a column 'name_img' containing image names.

    Returns:
    - pd.DataFrame: DataFrame with an additional column 'name_img_encode' representing the encoded image names.

    Example:
    ```python
    modified_dataframe = feature_engine(input_dataframe)
    ```

    The `feature_engine` function performs the following operations:
    1. Uses Label Encoding to encode the 'name_img' column in the input DataFrame (`df`).
    2. Adds a new column 'name_img_encode' to the DataFrame containing the encoded image names.

    Note:
    - Assumes that the input DataFrame has a column 'name_img' containing categorical image names.

    """
    le = LabelEncoder()
    le.fit(df["name_img"])
    df["name_img_encode"] = le.transform(df["name_img"])

    return df

def create_train_test_csv(df):
    """
    Splits a pandas DataFrame into training and testing sets and saves them as CSV files.

    Parameters:
    - df (pd.DataFrame): Input DataFrame to be split into training and testing sets.

    Example:
    ```python
    create_train_test_csv(input_dataframe)
    ```

    The `create_train_test_csv` function performs the following operations:
    1. Splits the input DataFrame (`df`) into training and testing sets based on row indices.
    2. Saves the training set as a CSV file named 'train_21.csv' in the specified datasets path (`v.DATASETS_PATH`).
    3. Saves the testing set as a CSV file named 'test_21.csv' in the specified datasets path.

    Note:
    - Assumes that the input DataFrame has been properly prepared and includes features and labels.
    - Assumes that the datasets path is correctly defined in the constants module or file.

    """
    df_train_comp = df.iloc[0:5465,:]
    df_test_comp = df.iloc[5465:,:]

    df_train_comp.to_csv(f"{v.DATASETS_PATH}\\train_21.csv", sep=";")
    df_test_comp.to_csv(f"{v.DATASETS_PATH}\\test_21.csv", sep=";")


