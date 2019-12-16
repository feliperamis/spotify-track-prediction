from preSVM import SVC, GridSearchCV
from preSVM import cross_val_score, plt, np, time
from preSVM import evaluate_classifier, save_output
from preSVM import x_train, y_train, x_test, y_test, K


# Interval for searching
# CS = np.logspace(-2, 6, num=9)
CS = np.logspace(-1, 5, num=7)
# Time
start = time()
# Cross - Validation
SVC_ = SVC(kernel="poly", degree=2)
param_grid = {'C': CS}
# K-fold Cross-Validation
grid_search = GridSearchCV(SVC_, param_grid, cv=K)
grid_search.fit(x_train, y_train)
# Plot the K-fold Cross-Validation accuracy depending on C
scores = grid_search.cv_results_["mean_test_score"]
plt.semilogx(CS, scores)
plt.show()
# Best C and Acc on training
parval = grid_search.best_params_
cvacc = cross_val_score(SVC(kernel="poly", degree=2, C=parval['C']), X=x_train, y=y_train, cv=K, scoring="accuracy")
# Train
SVM = SVC(kernel="poly", degree=2, C=parval['C'] , decision_function_shape='ovo')
SVM.fit(x_train, y_train)
# Evaluate and Save
output = evaluate_classifier(start, SVM, x_test, y_test, parval, cvacc)
save_output("outputs/polynomial.txt", output)
