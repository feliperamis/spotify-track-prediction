###############
##  Imports  ##
###############

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import itertools

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import model_selection as ms
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score


#################
##  Functions  ##
#################
def plot_confusion_matrix(base_confusion_matrix, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        base_confusion_matrix = base_confusion_matrix.astype('float') / base_confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(base_confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = base_confusion_matrix.max() / 2.
    for i, j in itertools.product(range(base_confusion_matrix.shape[0]), range(base_confusion_matrix.shape[1])):
        plt.text(j, i, format(base_confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if base_confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True digit')
    plt.xlabel('Predicted digit')
    plt.tight_layout()

def evaluate_classifier(start, clf, test_data, test_answers, parval, cvacc):
    test_predicted = clf.predict(test_data)
    acc_score = accuracy_score(test_answers, test_predicted)
    rec_score = recall_score(test_answers, test_predicted, average="macro")
    f_measure = 2 * acc_score * rec_score / (acc_score + rec_score)

    # Plot of confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix(test_answers, test_predicted), classes=[0, 1, 2], title='Confusion matrix of polynomial SVM')
    plt.show()

    output = f"Statistics {K}-fold cross\n"
    output += f"Elapsed time: {time() - start}\n"
    output += f"Train Data:\n\tAccuracy: {cvacc.mean()}\n"
    output += f"Test Data:\n\tConfusion Matrix:\n"
    for elem in confusion_matrix(test_answers, test_predicted):
        output += f"\t\t{elem}\n"
    output += f"\n\tAccuracy: {acc_score}\n"
    output += f"\tRecall/Sensitivity: {rec_score}\n"
    output += f"\tF - Measure: {f_measure}\n" 
    output += f"\tBest value of parameters found: {parval}\n"
    output += f"\tNumber of supports: {np.sum(clf.n_support_)} ({np.sum(np.abs(clf.dual_coef_) == parval['C'])} of them have slacks)\n"
    output += f"\tPorp. of supports: {np.sum(clf.n_support_) / test_data.shape[0]}\n"

    print(output)
    return output

def save_output(path, output):
    with open(path, 'w') as out:
        out.write(output)


#######################################
##  Open File & Final Preprocessing  ##
#######################################

PATH = "../../datasets/SpotifyDataset.csv"
df = pd.read_csv(PATH, header=0)
# Test with only 2 classes
# data['popularity'] = pd.qcut(data['popularity'], 2,labels=[0,1])

# Delete first column
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)], axis = 1, inplace = True)
# Pop popularity
x = df.drop(['popularity'], axis=1).values
y = df['popularity'].values
# Normalize data
min_max_scaler = preprocessing.MinMaxScaler()
x_norm = min_max_scaler.fit_transform(x)
# 30% of data for testing
(x_train, x_test, y_train, y_test) = ms.train_test_split(x_norm, y, test_size=0.3, stratify=y, random_state=1)

#####################
##  SVM Variables  ##
#####################

K = 10
