import imagePreprocessing
import kFold
import model
import confusionMatrix

import torchvision.models as models

def test(
    composed_dataset_path: str,
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
    modelType: str,
    accuracy_folder_path: str,
    loss_folder_path: str,
    feature_type_acc: str,
    feature_type_loss: str,
    results_file_path: str):
    
    #image preprocessing
    class_names, x, y = imagePreprocessing.preprocess_imagesv2(
        dataset_path=composed_dataset_path, 
        width=224, 
        height=224, 
        num_classes=num_classes, 
        framework="torch", 
        imagenet=True
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
        mode="feature_composer",
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
        mode="feature_composer", 
        num_classes = num_classes // 2,
        results_file_path = results_file_path,
        class_names = class_names
    )