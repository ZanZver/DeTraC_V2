'''
import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
'''
import random
import shutil
                
def execute_newInitaial_dataset(
    initial_dataset_path: str, 
    new_initial_dataset_path: str):

    # Check if folders exist
    assert os.path.exists(initial_dataset_path)
    assert os.path.exists(new_initial_dataset_path)
    
    initialFolders = os.listdir(initial_dataset_path)
    if('.DS_Store' in initialFolders):
        initialFolders.remove('.DS_Store')
    
    newInitialFolder = os.listdir(new_initial_dataset_path)
    if('.DS_Store' in newInitialFolder):
        newInitialFolder.remove('.DS_Store')

    for initialFolder in initialFolders:
        #print(initialFolder)
        initialFolderTmp = os.path.join(initial_dataset_path, initialFolder)
        #print(initialFolderTmp)
        imagesInInitialFolder = os.listdir(initialFolderTmp)
        if('.DS_Store' in imagesInInitialFolder):
            imagesInInitialFolder.remove('.DS_Store')
        random.shuffle(imagesInInitialFolder)
        #print(imagesInInitialFolder)
        firstImageSet = imagesInInitialFolder[:int(len(imagesInInitialFolder)/2)]
        secondImageSet = imagesInInitialFolder[int(len(imagesInInitialFolder)/2):]

        for file_name in firstImageSet:
            for item in newInitialFolder:
                if((initialFolderTmp.endswith(item[:-2])) and (item.endswith("1"))):
                    tempFirstFolder = os.path.join(new_initial_dataset_path, item)
                    shutil.copy(os.path.join(initialFolderTmp, file_name), tempFirstFolder)
                    
        for file_name in secondImageSet:
            for item in newInitialFolder:            
                if((initialFolderTmp.endswith(item[:-2])) and (item.endswith("2"))):
                    tempFirstFolder = os.path.join(new_initial_dataset_path, item)
                    shutil.copy(os.path.join(initialFolderTmp, file_name), tempFirstFolder)


    
    