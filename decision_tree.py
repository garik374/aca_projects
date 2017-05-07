from decision_tree_builder import build_tree
from decision_tree_builder import print_tree
import time

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
        # TODO: Build a tree that has self.max_depth
        # TODO: Remove this toto and the todo above after you
        # implement the todo above.
#        start_time = time.time()

        self.tree=build_tree((np.concatenate((X, Y), axis=1)), 0, self.max_depth)
       
    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        # TODO: Evaluate label of all the elements in `X` and
        # return same size list with labels.
        
        Y=[]
        for i in range (0, len(X)):
            tree=self.tree
            while tree.is_leaf==False:
                if type(X[i][tree.column]) is int or type(X[i][tree.column]) is float or type(X[i][tree.column]) is np.float64:
                    if X[i][tree.column]>tree.value:
                        tree=tree.true_branch
                    else:
                        tree=tree.false_branch
                else:
                    print("error")
                    if X[i][tree.column]==tree.value:
                        tree=tree.true_branch
                    else:
                        tree=tree.false_branch
            Y.append(tree.result)
        return Y
        
                
        


            
        # TODO: Remove this toto and the todo above after you
        # implement the todo above.
    
