from collections import defaultdict


class DecisionNode(object):
    """
    DecisionNode is a building block for Decision Trees.
    DecisionNode is a python class representing a  node in our decision tree
    node = DecisionNode()  is a simple usecase for the class
    you can also initialize the class like this:
    node = DecisionNode(column = 3, value = "Car")
    In python, when you initialize a class like this, its __init__ method is called 
    with the given arguments. __init__() creates a new object of the class type, and initializes its 
    instance attributes/variables.
    In python the first argument of any method in a class is 'self'
    Self points to the object which it is called from and corresponds to 'this' from Java

    """

    def __init__(self,
                 column=None,
                 value=None,
                 false_branch=None,
                 true_branch=None,
                 current_results=None,
                 is_leaf=False,
                 result=None):
        self.column = column
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.current_results = current_results
        self.is_leaf = is_leaf
        self.result = result


def dict_of_values(data):
    """
    param data: a 2D Python list representing the data. Last column of data is Y.
    return: returns a python dictionary showing how many times each value appears in Y

    for example 
    data = [[1,'yes'],[1,'no'],[1,'yes'],[1,'yes']]
    dict_of_values(data)
    should return {'yes' : 3, 'no' :1}
    """
    results = defaultdict(int)
    for row in data:
        r = row[len(row) - 1]
        results[r] += 1
    return dict(results)


def divide_data(data, feature_column, feature_val):
    """
    this function dakes the data and divides it in two parts by a line. A line
    is defined by the feature we are considering (feature_column) and the target 
    value. The function returns a tuple (data1, data2) which are the desired parts of the data.
    For int or float types of the value, data1 have all the data with values >= feature_val
    in the corresponding column and data2 should have rest.
    For string types, data1 should have all data with values == feature val and data2 should 
    have the rest.

    param data: a 2D Python list representing the data. Last column of data is Y.
    param feature_column: an integer index of the feature/column.
    param feature_val: can be int, float, or string
    return: a tuple of two 2D python lists
    """
    data1 = [data_inst for data_inst in data if data_inst[feature_column] >= feature_val]
    data2 = [data_inst for data_inst in data if data_inst[feature_column] < feature_val]

    return data1, data2


def gini_impurity(data1, data2):
    """
    Given two 2D lists of compute their gini_impurity index. 
    Remember that last column of the data lists is the Y
    Lets assume y1 is y of data1 and y2 is y of data2.
    gini_impurity shows how diverse the values in y1 and y2 are.
    gini impurity is given by 

    N1*sum(p_k1 * (1-p_k1)) + N2*sum(p_k2 * (1-p_k2))

    where N1 is number of points in data1
    p_k1 is fraction of points that have y value of k in data1
    same for N2 and p_k2


    param data1: A 2D python list
    param data2: A 2D python list
    return: a number - gini_impurity 
    """
    pk1_dic = defaultdict(lambda: 0.)
    pk2_dic = defaultdict(lambda: 0.)

    # data1
    for data_inst in data1:
        pk1_dic[data_inst[len(data_inst) - 1]] += 1
    for key, value in pk1_dic.items():
        pk1_dic[key] = value / len(data1)

    # data2
    for data_inst in data2:
        pk2_dic[data_inst[len(data_inst) - 1]] += 1
    for key, value in pk2_dic.items():
        pk2_dic[key] = value / len(data2)

    first_sum = len(data1) * sum([pk1 * (1 - pk1) for pk1 in pk1_dic.values()])
    second_sum = len(data2) * sum([pk2 * (1 - pk2) for pk2 in pk2_dic.values()])
    return first_sum + second_sum


def build_tree(data, current_depth=0, max_depth=1e10):
    """
    build_tree is a recursive function.
    What it does in the general case is:
    1: find the best feature and value of the feature to divide the data into
    two parts
    2: divide data into two parts with best feature, say data1 and data2
        recursively call build_tree on data1 and data2. this should give as two 
        trees say t1 and t2. Then the resulting tree should be 
        DecisionNode(...... true_branch=t1, false_branch=t2) 


    In case all the points in the data have same Y we should not split any more, and return that node
    For this function we will give you some of the code so its not too hard for you ;)
    
    param data: data. A 2D python list
    param current_depth: an integer. This is used if we want to limit the number of layers in the
        tree
    param max_depth: an integer - the maximal depth of the representing
    return: an object of class DecisionNode

    """
    if len(data) == 0:
        return DecisionNode(is_leaf=True)

    klass_proportion = dict_of_values(data)
    maximum = max(klass_proportion, key=klass_proportion.get)
    result = maximum

    if (current_depth == max_depth):
        return DecisionNode(current_results=dict_of_values(data), result=result, is_leaf=True)

    if (len(dict_of_values(data)) == 1):
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True, result=result)

    # This calculates gini number for the data before dividing
    self_gini = gini_impurity(data, [])

    # Below are the attributes of the best division that you need to find.
    # You need to update these when you find a division which is better
    best_gini = 1e10
    best_column = None
    best_value = None
    # best_split is tuple (data1,data2) which shows the two datas for the best divison so far
    best_split = None

    for feature_column in range(len(data[0]) - 1):
        sorted_features = sorted(set([data[i][feature_column] for i in range(len(data))]))
        lines = [(sorted_features[i] + sorted_features[i + 1]) / 2. for i in range(len(sorted_features) - 1)]
        for line in lines:
            data1, data2 = divide_data(data, feature_column, line)
            current_gini = gini_impurity(data1, data2)
            if current_gini < best_gini:
                best_gini = current_gini
                best_column = feature_column
                best_split = data1, data2
                best_value = line

    # if best_gini is no improvement from self_gini, we stop and return a node.
    if abs(self_gini - best_gini) < 1e-10:
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True, result=result)
    else:

        # recursively call build tree, construct the correct return argument and return
        return DecisionNode(column=best_column,

                            true_branch=build_tree(best_split[0], current_depth=current_depth + 1, max_depth=max_depth),

                            false_branch=build_tree(best_split[1], current_depth=current_depth + 1,
                                                    max_depth=max_depth),

                            current_results=dict_of_values(data), value=best_value, result=result)


# In[84]:

def print_tree(tree, indent=''):
    # Is this a leaf node?
    if tree.is_leaf:
        print(tree.result)
    else:
        # Print the criteria
        # print (indent+'Current Results: ' + str(tree.current_results))
        print(indent + 'Column ' + str(tree.column) + ' : ' + str(tree.value) + '? ')

        # Print the branches
        print(indent + 'True->', end="")
        print_tree(tree.true_branch, indent + '  ')
        print(indent + 'False->', end="")
        print_tree(tree.false_branch, indent + '  ')
