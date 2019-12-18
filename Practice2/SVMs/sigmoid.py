from preSVM import SVC, GridSearchCV
from preSVM import cross_val_score, plt, np, time
from preSVM import evaluate_classifier, save_output, plot_confusion_matrix
from preSVM import x_train, y_train, x_test, y_test, K


# Interval for searching
# CS = np.logspace(-2, 6, num=9)
CS = np.logspace(-1, 5, num=7)
# Time
start = time()
# Cross - Validation
SVC_ = SVC(kernel = 'sigmoid')
param_grid = {'C': CS, "gamma": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]} # "coef0": [0.0, 0.5, 1.0]
# K-fold Cross-Validation
grid_search = GridSearchCV(SVC_, param_grid, cv=K)
grid_search.fit(x_train, y_train)
# Best C
parval = grid_search.best_params_
# Plot accuracy for each combination of parameters tester
scores = grid_search.cv_results_["mean_test_score"]
scores = np.array(scores).reshape(len(param_grid['C']), len(param_grid["gamma"]))
plt.matshow(scores)
plt.xlabel("gamma")
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(param_grid["gamma"])), param_grid["gamma"], rotation="vertical")
plt.yticks(np.arange(len(param_grid['C'])), param_grid['C'])
plt.show()


# Acc on training
cvacc = cross_val_score(SVC(C=parval['C'], gamma=parval["gamma"]) , X=x_train,  y=y_train, cv=10, scoring="accuracy")
# Train
SVM = SVC(C=parval['C'], gamma=parval["gamma"]) 
SVM.fit(x_train, y_train)
# Evaluate and Save
output = evaluate_classifier(start, SVM, x_test, y_test, parval, cvacc)
save_output("outputs/rbf.txt", output)

# Plot of confusion matrix
plot_confusion_matrix(y_test, SVM.predict(y_test), 'popularity', 'cmatrix-sigmoid', 'Sigmoid matrix - 3 types of popularity')