import numpy as np
from sklearn.metrics import confusion_matrix
import multiclassConfusionMatrix

def test():
    print("confusion matrix")
    
def compose_classes(
    cmat: np.ndarray, 
    block_size: tuple):

    sizes = list(tuple(np.array(cmat.shape) // block_size) + block_size)
    for i in range(len(sizes)):
        if (i + 1) == len(sizes) - 1:
            break
        if i % 2 != 0:
            temp = sizes[i]
            sizes[i] = sizes[i + 1]
            sizes[i + 1] = temp

    reshaped_matrix = cmat.reshape(sizes)
    composed = reshaped_matrix.sum(axis=(1, 3))
    return composed
    
def compute_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    framework: str, 
    mode: str, 
    num_classes: int):

    # Check if a framework is chosen
    assert framework == "tf" or framework == "torch"

    # Check if a mode is selected
    assert mode == "feature_extractor" or mode == "feature_composer"
    
    # Create confusion matrix and normalize it
    cmat = confusion_matrix(
        y_true=y_true.argmax(axis=1), 
        y_pred=y_pred.argmax(axis=1), 
        normalize="all"
    )
    
    # If the feature composer was selected, divide the confusion matrix by NxN kernels
    if mode == "feature_composer":
        cmat = compose_classes(cmat, (2, 2))
        
    print(cmat)

    # Compute accuracy, sensitivity and specificity
    acc, sn, sp = multiclassConfusionMatrix.multiclass_confusion_matrix(cmat, num_classes)
    output = f"ACCURACY = {acc}"
    print(output)