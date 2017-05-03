import decision_tree_builder as tb
import numpy as np


class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of
        the trees.
    """

    def __init__(self, num_trees, max_tree_depth, ratio_per_tree=0.3):
        self.num_trees = num_trees
        self.max_depth = max_tree_depth
        self.ratio_per_tree = ratio_per_tree
        self.trees = None

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        # TODO:Build self.num_trees trees of depth self.max_tree_depth
        # with randomized data.
        # TODO: Remove this toto and the todo above after you
        # implement the todo above.

        self.trees = []
        data = np.hstack((X, Y))
        N = len(data)
        for _ in range(self.num_trees):
            index = np.arange(N)
            np.random.shuffle(index)
            tree = tb.build_tree(list(data[index[1:int(self.ratio_per_tree * N)]]), current_depth=0,
                                 max_depth=self.max_depth)
            self.trees.append(tree)

    def predict(self, X):
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
        Y = []
        conf = []
        for data_item in X:
            zero_count = 0
            one_count = 0
            for tree in self.trees:
                current_node = tree
                while not current_node.is_leaf:
                    if data_item[current_node.column] >= current_node.value:
                        current_node = current_node.true_branch
                    else:
                        current_node = current_node.false_branch
                if (current_node.result == 0):
                    zero_count += 1
                else:
                    one_count += 1
            if (zero_count > one_count):
                Y.append(0)
                conf.append(zero_count / (zero_count + one_count))
            else:
                Y.append(1)
                conf.append(one_count / (zero_count + one_count))

        return (Y, conf)
