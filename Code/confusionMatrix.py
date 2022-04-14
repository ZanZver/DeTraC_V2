from unittest import result
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import multiclassConfusionMatrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, plot_precision_recall_curve


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
    num_classes: int,
    results_file_path: str):

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
       
    # Compute accuracy, sensitivity and specificity
    acc, sn, sp = multiclassConfusionMatrix.multiclass_confusion_matrix(cmat, num_classes)
    
    
    output = f"ACCURACY = {acc}"
    print(output)
    
    from sklearn.metrics import precision_recall_fscore_support as score

    #predicted = [1,2,3,4,5,1,2,1,1,4,5] 
    #y_test = [1,2,3,4,5,1,2,1,1,4,1]

    precision, recall, fscore, support = score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='micro')

    results = str("""precision: """ + str(format(precision, '.3f')) + """
recall: """ + str((format(recall, '.3f'))) + """
fscore: """ + str((format(fscore, '.3f'))))
    
    print(results)
    f = open(results_file_path, "w")
    f.write(results)
    f.close