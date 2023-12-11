from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from libsvm.svmutil import *
import sys
import time

f = open("E_val", "w")

# Generate some random data for demonstration purposes
y_svm, X_svm = svm_read_problem('train.txt')
X_svm = np.array(X_svm)
y_svm = np.array(y_svm)
X = []
y = []
for i in range(len(X_svm)):
    X.append([])
    for j in range(len(X_svm[i])):
        X[i].append(X_svm[i][j + 1])
    y.append(y_svm[i])
repeat = 1

n_estimators = [100]
learning_rate = [0.1]
max_depth = [10]
n_estimators_length = max(len(str(n_estimators[i])) for i in range(len(n_estimators)))
learning_rate_length = max(len(str(learning_rate[i])) for i in range(len(learning_rate)))
max_depth_length = max(len(str(max_depth[i])) for i in range(len(max_depth)))

E_val = np.zeros((len(n_estimators), len(learning_rate), len(max_depth)))

now = time.time()

for i in range(len(n_estimators)):
    for j in range(len(learning_rate)):
        for k in range(len(max_depth)):
            for l in range(repeat):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=l * l + 1)
                # Create a GradientBoostingRegressor

                params = {
                    'n_estimators': n_estimators[i],  # Number of boosting stages to be run
                    'learning_rate': learning_rate[j],  # Step size shrinkage used to prevent overfitting
                    'max_depth': max_depth[k],  # Maximum depth of the individual trees
                    'random_state': l + 1,  # Seed for reproducibility
                    'subsample': 0.8,
                }

                gb_regressor = GradientBoostingRegressor(**params)

                # Fit the regressor to the training data
                gb_regressor.fit(X_train, y_train)

                # Make predictions on the test set
                y_pred = gb_regressor.predict(X_test)

                # Evaluate the model
                print("n_estimators, learning_rate, max_depth, time = " + str(n_estimators[i]).rjust(n_estimators_length) + ", " + str(learning_rate[j]).rjust(learning_rate_length) + ", " + str(max_depth[k]).rjust(max_depth_length) + ", " + str(time.time() - now), file=sys.stderr)
                now = time.time()
                E_val[i][j][k] += mean_squared_error(y_test, y_pred)
            E_val[i][j][k] /= repeat
            print("n_estimators, learning_rate, max_depth, E_val = " + str(n_estimators[i]).rjust(n_estimators_length) + ", " + str(learning_rate[j]).rjust(learning_rate_length) + ", " + str(max_depth[k]).rjust(max_depth_length) + ", " + str(E_val[i][j][k]), file=f)
            f.flush()