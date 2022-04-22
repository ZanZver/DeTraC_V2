from unittest import result
import numpy as np
from sklearn.metrics import confusion_matrix
import multiclassConfusionMatrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, plot_precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support as score

import pandas as pd
import seaborn
import matplotlib.pyplot as plt

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
    results_file_path: str,
    class_names):

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
    
    cmat2 = pd.crosstab(
        y_true.argmax(axis=1), 
        y_pred.argmax(axis=1), 
        normalize="all"
    )
    #print(num_classes)
    #print(cmat2)
    # If the feature composer was selected, divide the confusion matrix by NxN kernels
    if mode == "feature_composer":
        try:
            cmat = compose_classes(cmat, (2, 2))
        except:
            print("error")
       
    #print(cmat)
    #plt.figure()
    #cmatPlot = seaborn.heatmap(cmat, annot=True)
    #fig = cmatPlot.get_figure()
    #fig.savefig(str("/Users/zanzver/Documents/BCU2/Year_3/CMP6200:DIG6200-A-S1:S2-2021:2022_Individual_Undergraduate_Project/FYP/research/Code/DeTraCv2/DeTraC_V2/figure.png"))

       
    # Compute accuracy, sensitivity and specificity
    #acc, sn, sp = multiclassConfusionMatrix.multiclass_confusion_matrix(cmat, num_classes)
    
    #output = f"ACCURACY = {acc}"
    #print(output)

    #precision, recall, fscore, support = score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='micro')
    
    #print(results)
    #f = open(results_file_path, "w")
    #f.write(results)
    #f.close
    
    #pre = precision_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='macro') #[None, 'micro', 'macro', 'weighted']
    #re = recall_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
    #f1 = f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
    
    #results2 = str("""precision: """ + str(format(pre, '.3f')) + """recall: """ + str((format(re, '.3f'))) + """fscore: """ + str((format(f1, '.3f'))))
    #print(results2)
    
    confusion = cmat
    ConfusionMatrix= str('Confusion Matrix\n')
    ConfusionMatrixData= str(confusion)
    
    Accuracy = str('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))))

    MicroPrecision = str('Micro Precision: {:.2f}\n'.format(precision_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='micro')))
    MicroRecall = str('Micro Recall: {:.2f}\n'.format(recall_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='micro')))
    MicroF1score = str('Micro F1-score: {:.2f}\n'.format(f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='micro')))

    MacroPrecision = str('Macro Precision: {:.2f}\n'.format(precision_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='macro')))
    MacroRecall = str('Macro Recall: {:.2f}\n'.format(recall_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='macro')))
    MacroF1score = str('Macro F1-score: {:.2f}\n'.format(f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='macro')))

    WeightedPrecision = str('Weighted Precision: {:.2f}\n'.format(precision_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')))
    WeightedRecall = str('Weighted Recall: {:.2f}\n'.format(recall_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')))
    WeightedF1score = str('Weighted F1-score: {:.2f}\n'.format(f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')))

    ClassificationReport = str('\nClassification Report\n')
    try:
        ClassificationReportData = str(classification_report(y_true.argmax(axis=1), y_pred.argmax(axis=1), target_names = class_names))
    except:
        ClassificationReportData = str(classification_report(y_true.argmax(axis=1), y_pred.argmax(axis=1)))
    
    result2 = str( ConfusionMatrix +
ConfusionMatrixData +
Accuracy +
MicroPrecision +
MicroRecall +
MicroF1score +
MacroPrecision +
MacroRecall +
MacroF1score +
WeightedPrecision +
WeightedRecall +
WeightedF1score + 
ClassificationReport +
ClassificationReportData
    )
    
    print(result2)
    f = open(results_file_path+"/Results.txt", "w")
    f.write(result2)
    f.close