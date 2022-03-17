import trainFeatureExtractor
import trainFeatureComposer
import modelPrediction
import torchvision.models as models
import os
import createNewInitaialDataset

print("==========================================================")
print("Starting code")
print("==========================================================")
#==========================================================
# Models:
resnext50_32x4d = "resnext50_32x4d"
# 1) 0.45 2) 0.45 (20 in Folder 1, 20 in folder 2)
resnext101_32x8d = "resnext101_32x8d"
# 1) 0.39 2) 0.55 (20 in Folder 1, 20 in folder 2)
resnet18 = "resnet18"
# 1) 0.50 2) 0.70 (20 in Folder 1, 20 in folder 2)
resnet34 = "resnet34"
# 1) 0.35 2) 0.5(20 in Folder 1, 20 in folder 2)
resnet50 = "resnet50"
# 1) 0.50 2) 0.55 (20 in Folder 1, 20 in folder 2)
resnet101 = "resnet101"
# 1) 0.55 2) 0.70 (20 in Folder 1, 20 in folder 2)
resnet152 = "resnet152"
# 1) 0.50 2) 0.60 (20 in Folder 1, 20 in folder 2)
wide_resnet50_2 = "wide_resnet50_2"
# 1) 0.50 2) 0.55 (20 in Folder 1, 20 in folder 2)
wide_resnet101_2 = "wide_resnet101_2"
# 1) 0.50 2) 0.70 (20 in Folder 1, 20 in folder 2)
vgg19 = "vgg19"
# 1) 0.49 2) 0.50 (20 in Folder 1, 20 in folder 2)

# Variables:
num_epochs = 8
batch_size = 32
feature_extractor_num_classes = 8
feature_composer_num_classes = 2 * feature_extractor_num_classes
folds = 2
feature_extractor_lr = 0.01
feature_composer_lr = 0.01
use_cuda = False
k = 2
modelType = resnet18
momentumValue = 0.9
dropoutValue = 0.5
lrDecrese = 0.85
#drop lr by 0.85 ever 5 epochs

#==========================================================
# Data path:
DATA_FOLDER_PATH = "Data"
INITIAL_DATASET_PATH = os.path.join(DATA_FOLDER_PATH, "initial_dataset")
EXTRACTED_FEATURES_PATH = os.path.join(DATA_FOLDER_PATH, "extracted_features")
COMPOSED_DATASET_PATH = os.path.join(DATA_FOLDER_PATH, "composed_dataset")
INITIAL_DATASET_PATH_V2 = os.path.join(DATA_FOLDER_PATH, "initial_dataset_v2")

# Paths where models are stored
GENERAL_MODELS_PATH = "Models"
TF_MODEL_DIR = os.path.join(GENERAL_MODELS_PATH, "tf")
TORCH_CKPT_DIR = os.path.join(GENERAL_MODELS_PATH, "torch")

# Graphs path:
GRAPHS_FOLDER_PATH = "Graphs"
ACCURACY_FOLDER_PATH = os.path.join(GRAPHS_FOLDER_PATH, "AccuracyGraphs")
LOSS_FOLDER_PATH = os.path.join(GRAPHS_FOLDER_PATH, "LossGraphs")

FEATURE_EXTRACTOR_ACCURACY_FOLDER_PATH = os.path.join(ACCURACY_FOLDER_PATH, "FeatureExtractor")
FEATURE_COMPOSER_COMPOSEDPATH_ACCURACY_FOLDER_PATH = os.path.join(ACCURACY_FOLDER_PATH, "FeatureComposerComposedPath")
FEATURE_COMPOSER_INITIALPATH_ACCURACY_FOLDER_PATH = os.path.join(ACCURACY_FOLDER_PATH, "FeatureComposerInitialPath")

FEATURE_EXTRACTOR_LOSS_FOLDER_PATH = os.path.join(LOSS_FOLDER_PATH, "FeatureExtractor")
FEATURE_COMPOSER_COMPOSEDPATH_LOSS_FOLDER_PATH = os.path.join(LOSS_FOLDER_PATH, "FeatureComposerComposedPath")
FEATURE_COMPOSER_INITIALPATH_LOSS_FOLDER_PATH = os.path.join(LOSS_FOLDER_PATH, "FeatureComposerInitialPath")

#==========================================================
'''
    To do: if Exception has been thrown, stop the code
'''
print("==========================================================")
print("Checking folders")
print("==========================================================")
def checkFolders(foldername):
    try:
        if os.path.exists(foldername):
            print(str(foldername) + " folder exists")
        else:
            print(str(foldername) + " folder does not exist, creating the folder")
            os.mkdir(foldername)
    except Exception as e:
        print("Exception: "+ str(e))

checkFolders(DATA_FOLDER_PATH)  
checkFolders(INITIAL_DATASET_PATH)
checkFolders(EXTRACTED_FEATURES_PATH)
checkFolders(COMPOSED_DATASET_PATH)
checkFolders(INITIAL_DATASET_PATH_V2)

checkFolders(GENERAL_MODELS_PATH)
checkFolders(TF_MODEL_DIR)
checkFolders(TORCH_CKPT_DIR)

checkFolders(GRAPHS_FOLDER_PATH)
checkFolders(ACCURACY_FOLDER_PATH)
checkFolders(LOSS_FOLDER_PATH)

checkFolders(FEATURE_EXTRACTOR_ACCURACY_FOLDER_PATH)
checkFolders(FEATURE_COMPOSER_COMPOSEDPATH_ACCURACY_FOLDER_PATH)
checkFolders(FEATURE_COMPOSER_INITIALPATH_ACCURACY_FOLDER_PATH)

checkFolders(FEATURE_EXTRACTOR_LOSS_FOLDER_PATH)
checkFolders(FEATURE_COMPOSER_COMPOSEDPATH_LOSS_FOLDER_PATH)
checkFolders(FEATURE_COMPOSER_INITIALPATH_LOSS_FOLDER_PATH)

#==========================================================
#Task 1 
#train - feature extractor
'''
    To do: if Exception has been thrown, stop the code
'''

print("==========================================================")
print("Train - feature extractor")
print("==========================================================")
trainFeatureExtractor.test( initial_dataset_path=INITIAL_DATASET_PATH,
                            extracted_features_path=EXTRACTED_FEATURES_PATH,
                            epochs=num_epochs,
                            batch_size=batch_size,
                            num_classes=feature_extractor_num_classes,
                            folds=folds,
                            lr=feature_extractor_lr,
                            momentumValue=momentumValue,
                            dropoutValue=dropoutValue,
                            lrDecrese=lrDecrese,
                            cuda=use_cuda,
                            ckpt_dir=TORCH_CKPT_DIR,
                            composed_dataset_path = COMPOSED_DATASET_PATH,
                            k = k,
                            modelType = modelType,
                            accuracy_folder_path = ACCURACY_FOLDER_PATH,
                            loss_folder_path = LOSS_FOLDER_PATH,
                            feature_type_acc = FEATURE_EXTRACTOR_ACCURACY_FOLDER_PATH,
                            feature_type_loss = FEATURE_EXTRACTOR_LOSS_FOLDER_PATH)

createNewInitaialDataset.execute_newInitaial_dataset(INITIAL_DATASET_PATH,INITIAL_DATASET_PATH_V2)

#==========================================================
#Task 2 - train model on subclasses 
#train feature composer 
'''
    To do: if Exception has been thrown, stop the code
'''

featureComposerOld = False

if(featureComposerOld == True):
    print("==========================================================")
    print("Train - feature composer - composed path")
    print("==========================================================")
    trainFeatureComposer.test(  composed_dataset_path=COMPOSED_DATASET_PATH,
                                epochs=num_epochs,
                                batch_size=batch_size,
                                num_classes=feature_composer_num_classes,
                                folds=folds,
                                lr=feature_composer_lr,
                                momentumValue=momentumValue,
                                dropoutValue=dropoutValue,
                                lrDecrese=lrDecrese,
                                cuda=use_cuda,
                                ckpt_dir=TORCH_CKPT_DIR,
                                modelType = modelType,
                                accuracy_folder_path = ACCURACY_FOLDER_PATH,
                                loss_folder_path = LOSS_FOLDER_PATH,
                                feature_type_acc = FEATURE_COMPOSER_COMPOSEDPATH_ACCURACY_FOLDER_PATH,
                                feature_type_loss = FEATURE_COMPOSER_COMPOSEDPATH_LOSS_FOLDER_PATH)
else:
    print("==========================================================")
    print("Train - feature composer - initial dataset")
    print("==========================================================")
    trainFeatureComposer.test(  composed_dataset_path=INITIAL_DATASET_PATH_V2,
                                epochs=num_epochs,
                                batch_size=batch_size,
                                num_classes=feature_composer_num_classes,
                                folds=folds,
                                lr=feature_composer_lr,
                                momentumValue=momentumValue,
                                dropoutValue=dropoutValue,
                                lrDecrese=lrDecrese,
                                cuda=use_cuda,
                                ckpt_dir=TORCH_CKPT_DIR,
                                modelType = modelType,
                                accuracy_folder_path = ACCURACY_FOLDER_PATH,
                                loss_folder_path = LOSS_FOLDER_PATH,
                                feature_type_acc = FEATURE_COMPOSER_INITIALPATH_ACCURACY_FOLDER_PATH,
                                feature_type_loss = FEATURE_COMPOSER_INITIALPATH_LOSS_FOLDER_PATH)

#==========================================================
#Task 3 - predict

print("==========================================================")
print("Predict")
print("==========================================================")
modelPrediction.test()

#==========================================================