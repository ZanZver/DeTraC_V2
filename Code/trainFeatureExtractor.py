import imagePreprocessing
import kFold
import model
import confusionMatrix
import featureExtraction
import decomposition

import torchvision.models as models
import numpy as np
import os

def test(initial_dataset_path: str,
    extracted_features_path: str,
    epochs: int,
    batch_size: int,
    num_classes: int,
    folds: int,
    lr:float,
    momentumValue:float,
    dropoutValue:float,
    lrDecrese:float,
    cuda: bool,
    ckpt_dir: str,
    composed_dataset_path: str,
    k,
    modelType,
    accuracy_folder_path: str,
    loss_folder_path: str,
    feature_type_acc: str,
    feature_type_loss: str):
    
    print(accuracy_folder_path)
    print(loss_folder_path)
    #image preprocessing
    class_names, x, y = imagePreprocessing.preprocess_imagesv2(
        dataset_path=initial_dataset_path, 
        width=224, 
        height=224, 
        num_classes=num_classes, 
        framework="torch"
    )
    
    #k-fold / cross validation
    X_train, X_test, Y_train, Y_test = kFold.KFold_cross_validation_splitv2(
        features=x, 
        labels=y, 
        n_splits=folds
    )
    
    # Normalize
    X_train /= 255
    X_test /= 255
    
    #model creation and training
    if(modelType == "resnext50_32x4d"):
        modelType = models.resnext50_32x4d(pretrained=True)
    elif(modelType == "resnext101_32x8d"):
        modelType = models.resnext101_32x8d(pretrained=True)
    elif(modelType == "resnet18"):
        modelType = models.resnet18(pretrained=True)
    elif(modelType == "resnet34"):
        modelType = models.resnet34(pretrained=True)
    elif(modelType == "resnet50"):
        modelType = models.resnet50(pretrained=True)
    elif(modelType == "resnet101"):
        modelType = models.resnet101(pretrained=True)
    elif(modelType == "resnet152"):
        modelType = models.resnet152(pretrained=True)
    elif(modelType == "wide_resnet50_2"):
        modelType = models.wide_resnet50_2(pretrained=True)
    elif(modelType == "wide_resnet101_2"):
        modelType = models.wide_resnet101_2(pretrained=True)
    elif(modelType == "vgg19"):
        modelType = models.vgg19(pretrained=True)   

    net = model.Net(
        modelType,
        num_classes=num_classes,
        lr=lr,
        cuda=cuda,
        mode="feature_extractor",
        ckpt_dir=ckpt_dir,
        labels=class_names,
        momentumValue=momentumValue,
        dropoutValue=dropoutValue,
        lrDecrese=lrDecrese
    )
  
    net.fitv2(
        X_train,
        Y_train,
        X_test,
        Y_test,
        epochs,
        batch_size,
        resume=False,
        accuracy_folder_path = accuracy_folder_path,
        loss_folder_path = loss_folder_path,
        feature_type_acc = feature_type_acc,
        feature_type_loss = feature_type_loss
    )
  
    #confusion matrix
    confusionMatrix.compute_confusion_matrix(
        y_true=Y_test, 
        y_pred=net.infer(X_test), 
        framework="torch", 
        mode="feature_extractor", 
        num_classes = num_classes)
    
    #feature extraction
    for class_name in class_names:
        extracted_features = featureExtraction.extract_features(
            initial_dataset_path=initial_dataset_path, 
            class_name=class_name, 
            width=224, 
            height=224, 
            net=net, 
            framework="torch"
        )
        
        np.save(
            file=os.path.join(extracted_features_path, f"{class_name}.npy"), 
            arr=extracted_features
        )

    #decomposition based on class names
    decomposition.execute_decomposition(initial_dataset_path,
                                        composed_dataset_path,
                                        extracted_features_path,
                                        k)