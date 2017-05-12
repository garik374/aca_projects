import numpy as np
from collections import defaultdict

"""
YOU MUST!!!
Read all the lines of the code provided to you and understand what it does!
"""


class DecisionNode(object):
    """
    README
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
    this function makes the data and divides it in two parts by a line. A line
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
    #TODO

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
    #TODO

    gini=0
    N1=len(data1)
    N2=len(data2)
    p1=[0,0,0]
    p2=[0,0,0]
    for j in range(3):
        for i in dict_of_values(data1).keys():
            p1[j]=dict_of_values(data1)[i]/N1
        for i in dict_of_values(data2).keys():
            p2[j]=dict_of_values(data2)[i]/N2
    for i in range(3):
        gini=gini+N1*p1[i]*(1-p1[i])
    for i in range(3):
        gini=gini+N2*p2[i]*(1-p2[i])
    return gini



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
    
    param data: param data: A 2D python list
    param current_depth: an integer. This is used if we want to limit the numbr of layers in the
        tree
    param max_depth: an integer - the maximal depth of the representing
    return: an object of class DecisionNode

    """
    
    if len(data) == 0:
        return DecisionNode(is_leaf=True)

    klass_proportion = dict_of_values(data)
    maximum = max(klass_proportion, key=klass_proportion.get)
    result = maximum

    if(current_depth == max_depth):
        return DecisionNode(current_results=dict_of_values(data), result=result, is_leaf=True)

    if(len(dict_of_values(data)) == 1):
        return DecisionNode(current_results=dict_of_values(data), result=result, is_leaf=True)

    #This calculates gini number for the data before dividing 
    self_gini = gini_impurity(data, [])

    #Below are the attributes of the best division that you need to find. 
    #You need to update these when you find a division which is better
    best_gini = 1e10
    best_column = None
    best_value = None
    #best_split is tuple (data1,data2) which shows the two datas for the best divison so far
    best_split = None

    #TODO
    #You need to find the best feature to divide the data
    #For each feature and each possible value of the feature compute the 
    # gini number for that division. You need to find the feature that minimizes
    # gini number. Remember that last column of data is Y
    # Think how you can use the divide_data and gini_impurity functions you wrote 
    # above

    for i in range(len(data[0])-1):
        for row in data:
            d1,d2=divide_data(data,i,row[i])
            gini=gini_impurity(d1, d2)
            if best_gini>gini:
                best_gini=gini
                best_column=i
                best_val=row[i]
                best_split=divide_data(data, best_column, best_val)

    #if best_gini is no improvement from self_gini, we stop and return a node.
    if abs(self_gini - best_gini) < 1e-10:
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True, result=result)
    else:
        #TODO
        #recursively call build tree, construct the correct return argument  and return
        current_depth=current_depth+1
        t=build_tree(best_split[0], current_depth, max_depth=1e10)
        f=build_tree(best_split[1], current_depth, max_depth=1e10)
        
        
        return DecisionNode(column=best_column, value=best_val, false_branch=f, true_branch=t, current_results=dict_of_values(data),result=result) #<---- FIX THIS



def print_tree(tree, indent=''):
    # Is this a leaf node?
    if tree.is_leaf:
        print(str(tree.current_results))
    else:
        # Print the criteria
        #         print (indent+'Current Results: ' + str(tree.current_results))
        print('Column ' + str(tree.column) + ' : ' + str(tree.value) + '? ')

        # Print the branches
        print(indent + 'True->', end="")
        print_tree(tree.true_branch, indent + '  ')
        print(indent + 'False->', end="")
        print_tree(tree.false_branch, indent + '  ')

