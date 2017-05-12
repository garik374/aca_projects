import numpy as np
from dtOmitted import build_tree
from dtOmitted import DecisionNode
from dtOmitted import print_tree
from decision_tree import predict_row
class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of point
    s to use to train each of
        the trees.
    """
    def __init__(self, max_tree_depth, num_trees=10, ratio_per_tree=0.5):
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.trees = None
        self.ratio_per_tree=ratio_per_tree
    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        # TODO:Build self.num_trees trees of depth self.max_tree_depth
        # with randomized data.
        # TODO: Remove this toto and the todo above after you
        # implement the todo above.
        
        
        X=np.array(X)
        Y=np.array(Y)
        Z=np.hstack((X,Y))
        n=int(self.ratio_per_tree*len(Z))
        self.trees=[]
        for _ in range(self.num_trees):
            np.random.shuffle(Z)
            V=Z[:n,:]
            z=build_tree(V,current_depth=0, max_depth=self.max_tree_depth)
            self.trees.append(z)
        
    def predict1(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """
        # TODO: Evaluate labels in each of the `self.tree`s and return the
        # label and confidence with the most votes for each of
        # the data points in `X`
        # TODO: Remove this toto and the todo above after you
        # implement the todo above.
        
        Y=[]
        for i in range(len(X)):
            V=[]
            for z in self.trees:
                V.append(predict_row(z,X[i]))
                #print_tree(z)
                #print(predict_row(z,X[i]))
            Y.append(max(set(V), key = list(V).count))
        #print(predict_tree(self.trees[1],X))
        

        return Y   
