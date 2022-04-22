from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os
from PIL import Image

def imageAumentationOnOneFolder(image_directory, folder_name):
    print("Augmenting images for folder: " + folder_name)
    datagen = ImageDataGenerator(        
            rotation_range = 40,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            brightness_range = (0.5, 1.5))

    SIZE = 224
    dataset = []
    my_images = os.listdir(image_directory)

    for i, image_name in enumerate(my_images):  
        if (image_name.endswith(".jpg") or image_name.endswith(".tif")):        
            image = io.imread(image_directory + image_name)        
            image = Image.fromarray(image, 'RGB')        
            image = image.resize((SIZE,SIZE)) 
            dataset.append(np.array(image))
            

    x = np.array(dataset)
    i = 0
    augmentedDir = 'Data/augmented/'+folder_name+"/"
    if os.path.exists(augmentedDir):
        os.rmdir(augmentedDir)
    os.makedirs(augmentedDir)
    for batch in datagen.flow(x, batch_size=16,
                            save_to_dir= augmentedDir,
                            save_prefix=folder_name,
                            save_format='tif'):    
        i += 1    
        if i > 50:        
            break

def augmentImages(dataFolder):
    for folder in os.listdir(dataFolder):
        if(folder != ".DS_Store"):
            #print(folder)
            imageFolder = dataFolder + "/" + folder + "/"
            #print(imageFolder)
            imageAumentationOnOneFolder(imageFolder, folder)