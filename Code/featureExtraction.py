from tqdm import tqdm
import os
import cv2
import tensorflow as tf
import torch
import numpy as np

import os

import cv2

import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix
#from .multiclass_confusion_matrix import multiclass_confusion_matrix

from tqdm import tqdm

def test():
    print("feature extraction")
    
def extract_features(
    initial_dataset_path: str, 
    class_name: str, 
    width: int, 
    height: int,
    net,
    framework: str):

    # Check if a framework is chosen
    assert framework == "tf" or framework == "torch"

    # Initialize features list
    features = []

    # Initialize progress bar for folder
    #print(os.listdir(os.path.join(initial_dataset_path, class_name)))
    progress_bar = tqdm(os.listdir(os.path.join(initial_dataset_path, class_name)))
    progress_bar.set_description(f"Preparing {class_name} for feature extraction") 

    # Iterate through files in directory
    for filename in progress_bar:
        if filename != ".DS_Store":
            # Read grayscale image
            gray_img = cv2.imread(os.path.join(initial_dataset_path, class_name, filename))

            # Convert to RGB
            color_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)

            # Resize to the required size
            img = cv2.resize(color_img, (width, height))

            # Convert image to array
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            # Preprocess images for pretrained model
            if framework == "torch":
                img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode="torch")
            else:
                img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode="tf")

            # Append image array to features list
            features.append(img)

    # Convert features list to vertical stack array
    features = np.vstack(features)

    # Return (Nx4096) prediction a.k.a extract features.
    return net.infer_using_pretrained_layers_without_last(features)