import numpy as np
import matplotlib.pyplot as plt

from decision_tree import DecisionTree

from random_forest import RandomForest


def accuracy_score(y_true, y_predict):
    true_prediction_num = sum([int(true_value == predic_value) for true_value, predic_value in zip(y_true, y_predict)])

    return true_prediction_num / len(y_true)


def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of logistic regression
      stats[1,1] = std deviation of logistic regression accuracy

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = np.array(data[:, 1:])
    y = np.array([data[:, 0]]).T
    n, d = X.shape
    all_accuracies_dt = []
    all_accuracies_rf = []
    for trial in range(10):
        # TODO: shuffle for each of the trials.
        # the following code is for reference only.
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # TODO: write your own code to split data (for cross validation)
        # the code here is for your reference.
        N = int(9 * len(X) / 10)
        xtrain = X[1:N, :]
        xtest = X[N:, :]
        ytrain = y[1:N, :]
        ytest = y[N:, :]

        # train the decision tree
        classifier_dt = DecisionTree(100)
        classifier_dt.fit(xtrain, ytrain)

        classifier_rf = RandomForest(5, 100)
        classifier_rf.fit(xtrain, ytrain)

        # output predictions on the remaining data
        y_pred_dt = classifier_dt.predict(xtest)
        accuracy_dt = accuracy_score(ytest, y_pred_dt)
        all_accuracies_dt.append(accuracy_dt)

        y_pred_rf, conf = classifier_rf.predict(xtest)
        accuracy_rf = accuracy_score(ytest, y_pred_rf)
        all_accuracies_rf.append(accuracy_rf)

    # compute the training accuracy of the model
    meanDecisionTreeAccuracy = np.mean(all_accuracies_dt)

    # TODO: update these statistics based on the results of your experiment
    stddevDecisionTreeAccuracy = np.std(all_accuracies_dt)
    meanRandomForestAccuracy = np.mean(all_accuracies_rf)
    stddevRandomForestAccuracy = np.std(all_accuracies_rf)
    meanLogisticRegressionAccuracy = 0
    stddevLogisticRegressionAccuracy = 0

    # make certain that the return value matches the API specification
    stats = np.zeros((3, 3))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanRandomForestAccuracy
    stats[1, 1] = stddevRandomForestAccuracy
    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluate_performance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
# ...to HERE.
