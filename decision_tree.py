import numpy as np
from dtOmitted import DecisionNode
from dtOmitted import build_tree


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

        data=np.hstack((X,Y))

        
  
        self.trees = build_tree(data, max_depth=self.max_depth)



    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        def pred(tree, row):
            if tree.true_branch == None:
                return tree.result
            
            if type(tree.value) == int or type(tree.value) == float or type(tree.value) == np.float64 or type(tree.value)==np.int64:
                if row[tree.column]>=tree.value:
                    return pred(tree.true_branch, row)
                else:
                    return pred(tree.false_branch, row)
                
            else:
                if row[tree.column]==tree.value:
                    return pred(tree.true_branch, row)
                else:
                    return pred(tree.false_branch, row)
           
        Y=[]
        for i in range(len(X)):
            Y.append(pred(self.trees, X[i]))
        
        return Y
    
def predict_row(tree, row):
    if tree.true_branch == None:
        return tree.result
            
    if type(tree.value) == int or type(tree.value) == float or type(tree.value) == np.float64 or type(tree.value)==np.int64:
        if row[tree.column]>=tree.value:
            return predict_row(tree.true_branch, row)
        else:
            return predict_row(tree.false_branch, row)
                
    else:
        if row[tree.column]==tree.value:
            return predict_row(tree.true_branch, row)
        else:
            return predict_row(tree.false_branch, row)
