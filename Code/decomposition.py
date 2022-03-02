import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt

def test():
    print("decomposition based on class names")
    
def decompose(
    path_to_features: str, 
    path_to_images: str, 
    path_to_decomposed_images_1: str, 
    path_to_decomposed_images_2: str, 
    class_name: str,
    k: int):

    # Load features
    features = np.load(path_to_features)

    # Cluster index
    idx = KMeans(n_clusters=k, random_state=0).fit(features)
    idx = idx.predict(features)

    # Images list
    images = [filename for filename in os.listdir(path_to_images)]
    #print(images)
    if ".DS_Store" in images: images.remove(".DS_Store")
    # Iterate through images
    progress_bar = tqdm(range(len(images)))
    progress_bar.set_description(f"Composing {class_name} images")
    for i in progress_bar:
        filename = os.path.join(path_to_images, images[i])
        # Read image§
        I = plt.imread(filename)
        filename_1 = os.path.join(path_to_decomposed_images_1, images[i])
        filename_2 = os.path.join(path_to_decomposed_images_2, images[i])
        
        # If image belongs to a cluster, write the image to a certain folder, otherwise, write it to the other folder.
        if (idx[i] == 1):
            plt.imsave(filename_1, I)
        else:
            plt.imsave(filename_2, I)
    
def execute_decomposition(
    initial_dataset_path: str, 
    composed_dataset_path: str, 
    features_path: str,
    k:int):

    # Check if folders exist
    assert os.path.exists(initial_dataset_path)
    assert os.path.exists(composed_dataset_path)
    assert os.path.exists(features_path)

    # Initialize list of classes
    class_names = []

    # Check if folders exist and add them to the list of classes
    for folder in os.listdir(initial_dataset_path):
        if folder != ".DS_Store":
            assert os.path.isdir(os.path.join(initial_dataset_path, folder))
            class_names.append(folder)

    # For every class
    for class_name in class_names:
        try:
            # Create folder for 1st cluster.
            os.mkdir(os.path.join(composed_dataset_path, f"{class_name}_1/"))
        except:
            # If it cannot create a folder, it will overwrite it.
            print(f"Directory {class_name}_1 already exists. Overwriting.")
            shutil.rmtree(os.path.join(composed_dataset_path, f"{class_name}_1/"))
            os.mkdir(os.path.join(composed_dataset_path, f"{class_name}_1/"))
        try:
            # Create folder for 2nd cluster
            os.mkdir(os.path.join(composed_dataset_path, f"{class_name}_2/"))
        except:
            # If it cannot create a folder, it will overwrite it.
            print(f"Directory {class_name}_2 already exists. Overwriting.")
            shutil.rmtree(os.path.join(composed_dataset_path, f"{class_name}_2/"))
            os.mkdir(os.path.join(composed_dataset_path, f"{class_name}_2/"))

        # Decompose said class' data
        decompose(
            path_to_features=os.path.join(features_path, f"{class_name}.npy"),
            path_to_images=os.path.join(initial_dataset_path, class_name),
            path_to_decomposed_images_1=os.path.join(composed_dataset_path, f"{class_name}_1/"),
            path_to_decomposed_images_2=os.path.join(composed_dataset_path, f"{class_name}_2/"),
            class_name=class_name,
            k=k
        )