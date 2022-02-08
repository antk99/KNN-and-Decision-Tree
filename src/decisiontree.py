import numpy as np
import pandas as pd

# TODO testing/experimenting on the datasets w/ visualizations & for correctness

"""
The following cost functions were taken from the class website code (colab):
    1 - cost_misclassification(labels)
    2 - cost_entropy(labels)
    3 - def cost_gini_index(labels)
"""


# computes misclassification cost by subtracting the maximum probability of any class
def cost_misclassification(labels):
    counts = np.bincount(labels)
    class_probs = counts / np.sum(counts)
    return 1 - np.max(class_probs)


# computes entropy of the labels by computing the class probabilities
def cost_entropy(labels):
    class_probs = np.bincount(labels) / len(labels)
    class_probs = class_probs[class_probs > 0]  # this steps is remove 0 probabilities for removing numerical issues while computing log
    return -np.sum(class_probs * np.log(class_probs))   # expression for entropy -\sigma p(x)log[p(x)]


# computes the gini index cost
def cost_gini_index(labels):
    class_probs = np.bincount(labels) / len(labels)
    return 1 - np.sum(np.square(class_probs))   # expression for gini index 1-\sigma p(x)^2


def merge_features_labels(features, labels):
    """
    Helper function that re-merges X_train & Y_train into one DataFrame data_train
    :param features: Numpy.ndarray - X_train
    :param labels: Numpy.ndarray - Y_train
    :return: Pandas.DataFrame - data_train
    """
    data = pd.DataFrame(features)
    labels = pd.DataFrame(labels)
    data['Labels'] = labels
    return data


def best_split(data, cost_function):
    """
    Determines the best split of the data by iterating through all possible split values & calculates the cost of that
    split using the given parameter cost_function. The best split is the split with the lowest cost.
    :param data: Pandas.DataFrame - the data to be split
    :param cost_function: function - the cost function of the decision tree that computes a split cost
    :return: dict - {'cost': best split cost, 'feature': best split feature, 'test': best split value,
                     'data': {'left': left split data, 'right': right split data}}
    """

    # initializes the dictionary storing the information of the best data split to be returned
    split = {'cost': np.inf, 'feature': None, 'test': None, 'data': {'left': None, 'right': None}}

    # iterates through each possible feature column in the data (-1 because of assumed last column being the labels)
    for feature_index in range(data.shape[1] - 1):
        feature_vector = data.values[:, feature_index]  # stores the column of the feature at feature_index
        feature_tests = np.unique(feature_vector)   # stores all unique values for that feature

        # iterates through each possible value for a test on the current feature
        for test in feature_tests:
            # split the data into left and right children
            data_left = data.loc[data[data.columns[feature_index]] <= test]  # stores the data that is <= the curr test
            data_right = data.loc[data[data.columns[feature_index]] > test]  # stores the data that is > the curr test

            if data_left.shape[0] >= 0 and data_right.shape[0] >= 0:
                cost_left = cost_function(data_left.iloc[:, -1])    # computes cost of the left split
                cost_right = cost_function(data_right.iloc[:, -1])  # computes cost of the right split

                # computes the total cost of the split by weighing the left and right costs accordingly
                cost_weighted = ((data_left.shape[0] * cost_left) + (data_right.shape[0] * cost_right)) / data.shape[0]

                # if the current total cost is lower than our best cost found so far --> update the best cost to the
                # current split
                if cost_weighted < split['cost']:
                    split = {'cost': cost_weighted, 'feature': feature_index, 'test': test,
                             'data': {'left': data_left, 'right': data_right}}
    return split


class Node:
    """
    Stores information contained at a node from the decision tree
    """
    def __init__(self, feature=None, test=None, left_child=None, right_child=None, label=None):
        """
        Initializes a Node object & sets the given parameters
        :param feature: int - the index of the feature from the dataset that this node will test on
        :param test: the split value on the feature stored in self.feature
        :param left_child: Node - the left child of this node. Can be None if self.is_leaf()
        :param right_child: Node - the right child of this node. Can be None if self.is_leaf()
        :param label: int - if self.is_leaf(), it stores the leaf's label. None otherwise.
        """
        self.feature = feature
        self.test = test
        self.left_child = left_child
        self.right_child = right_child
        self.label = label

    def is_leaf(self):
        """
        Determines if self is a leaf node in the decision tree by checking if self has no children
        :return: True if self is a leaf node & False otherwise
        """
        return self.left_child is None and self.right_child is None

    def get_label(self):
        """
        Checks if self is a leaf node & gives the label if so
        :return: label stored in this leaf node, None otherwise
        """
        if self.is_leaf():
            return self.label


class DecisionTree:
    """
    Decision Tree algorithm is implemented as follows:

        Calling DecisionTree.fit() will build the decision tree using the private DecisionTree._build():
            1 - if there are enough instances in the data and not passed the max depth of the tree,
                it splits the data according to the greedy split function best_split() which iterates through
                all possible split values & calculates the cost of that split using the given parameter cost_function.
                The function returns the split with the minimal cost.
            2 - it, recursively, builds the left subtree and right subtree from the current node following step 1 on
                every new node until it reaches a leaf node. At leaf nodes, only the label is stored in the node.
                This is the classification class when using the decision tree.

        Calling DecisionTree.predict() will, for each instance in the test data, go through each node of the decision
        tree following the split test values to go to the left/right child for each node until it reaches a leaf node
        at which that instance's label is determined.
    """
    def __init__(self, max_depth=3, min_instances=1, cost_function=cost_gini_index):
        """
        Initializes the decision tree's root as None and stores the given (optional) parameters
        :param max_depth: int - the maximum depth of the tree
        :param min_instances: int - the minimum number of instances required to split a node
        :param cost_function: function - a cost function that takes in labels & computes the split cost
        """
        self.root = None
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.cost_function = cost_function

    def _build(self, data, curr_depth=0):
        """
        Recursively builds the entire decision tree given the data as follows:
            1 - if there are enough instances in the data and not passed the max depth of the tree,
                it splits the data according to the greedy split function best_split()
            2 - it, recursively, builds the left subtree and right subtree from the current node following step 1 on
                every new node until it reaches a leaf node. At leaf nodes, only the label is stored in the node.
                Leaf nodes are the classification class when using the decision tree.
        :param data: Pandas.DataFrame - the training data to build our decision tree on
        :param curr_depth: int - the current depth of the tree
        :return: Node
        """
        features = data.values[:, :-1]          # stores the 2D array of features (assumes last col are labels)
        num_instances = features.shape[0]       # stores number of instances of the data (num of rows)

        # if there are enough instances in the data and not passed the max depth of the tree --> split data
        if num_instances >= self.min_instances and curr_depth <= self.max_depth:
            split = best_split(data, self.cost_function)             # gets the split with minimum cost (best split)

            if np.isinf(split['cost']) or split['feature'] is None:  # this shouldn't happen w/ reasonable min_instances
                return

            # recursively builds the left and right subtrees of the current node
            left_child = self._build(split['data']['left'], curr_depth+1)
            right_child = self._build(split['data']['right'], curr_depth+1)

            return Node(feature=split['feature'], test=split['test'], left_child=left_child, right_child=right_child)

        # reached a leaf node so the label of this node is determined by the most frequent label in the data (mode)
        leaf_label = data['Labels'].mode().values
        return Node(label=leaf_label)

    def fit(self, train_features, train_labels):
        """
        Builds the decision tree with the given training data
        :param train_features: Numpy.ndarray - the training features (X_train)
        :param train_labels: Numpy.ndarray - the training labels (Y_train)
        :return: DecisionTree - the tree that was just built, self
        """
        self.root = self._build(merge_features_labels(train_features, train_labels))
        return self

    def predict(self, test_data):
        """
        For each instance in the test data, it goes through each node of the decision tree following the split test
        values to go to the left/right child for each node until it reaches a leaf node at which that instance's label
        is determined.
        :param test_data: Pandas.DataFrame - the test data for which it will classify each instance
        :return: Numpy.ndarray - the array of predictions for each instance in the test data
        """
        predictions = np.zeros(test_data.shape[0])      # initializes the array of predictions as zeros

        # iterates through each instance and determines that instance's label by following the splits of the dec tree
        for index, instance in enumerate(test_data):
            curr_node = self.root
            while not curr_node.is_leaf():
                if instance[curr_node.feature] <= curr_node.test:
                    curr_node = curr_node.left_child
                else:
                    curr_node = curr_node.right_child

            # at end of the while loop for each instance, curr_node stores the leaf node that has the predicted label
            # for the instance, so stores that in the array of predictions for that instance
            predictions[index] = curr_node.get_label()
        return predictions
