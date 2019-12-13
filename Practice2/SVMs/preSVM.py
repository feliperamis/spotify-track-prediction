###############
##  Imports  ##
###############

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import model_selection as ms
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score


#################
##  Functions  ##
#################

def evaluate_classifier(start, clf, test_data, test_answers, parval, cvacc):
    test_predicted = clf.predict(test_data)
    acc_score = accuracy_score(test_answers, test_predicted)
    rec_score = recall_score(test_answers, test_predicted, average="macro")
    f_measure = 2 * acc_score * rec_score / (acc_score + rec_score)

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
