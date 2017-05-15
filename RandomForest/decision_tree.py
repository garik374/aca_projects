import decision_tree_builder as tb
import numpy as np


class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """

    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """

        data = np.hstack((X, Y))
        self.trees = tb.build_tree(list(data), current_depth=0, max_depth=self.max_depth)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y : 1 dimension python list with labels
        """

        Y = []
        for data_item in X:
            current_node = self.trees
            while current_node.is_leaf == False:
                if data_item[current_node.column] >= current_node.value:
                    current_node = current_node.true_branch
                else:
                    current_node = current_node.false_branch
            Y.append(current_node.result)
        return Y
