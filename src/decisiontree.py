import numpy as np
import pandas as pd

# TODO comment/document code & cleaning of code
# TODO finish implementing entropy & missclassification cost functions
# TODO testing/experimenting on the datasets w/ visualizations & for correctness


# computes the gini index cost
def cost_gini_index(labels):
    class_probs = np.bincount(labels) / len(labels)
    return 1 - np.sum(np.square(class_probs))               # expression for gini index 1-\sigma p(x)^2


def merge_features_labels(features, labels):
    data = pd.DataFrame(features)
    labels = pd.DataFrame(labels)
    data['Labels'] = labels
    return data


class Node:
    def __init__(self, feature=None, test=None, left_child=None, right_child=None, label=None):
        self.feature = feature
        self.test = test
        self.left_child = left_child
        self.right_child = right_child
        self.label = label

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    def get_label(self):
        if self.is_leaf():
            return self.label


class DecisionTree:
    def __init__(self, max_depth=3, min_instances=1, cost_function=cost_gini_index):
        self.root = None
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.cost_function = cost_function

    def _build(self, data, curr_depth=0):
        features = data.values[:, :-1]
        num_instances = features.shape[0]

        if num_instances >= self.min_instances and curr_depth <= self.max_depth:
            split = best_split(data, self.cost_function)
            if np.isinf(split['cost']) or split['feature'] is None:
                return

            left_child = self._build(split['data']['left'], curr_depth+1)
            right_child = self._build(split['data']['right'], curr_depth+1)

            return Node(feature=split['feature'], test=split['test'], left_child=left_child, right_child=right_child)

        leaf_label = data['Labels'].mode().values
        return Node(label=leaf_label)

    def fit(self, train_features, train_labels):
        self.root = self._build(merge_features_labels(train_features, train_labels))

    def predict(self, test_data):
        predictions = []

        for index, instance in enumerate(test_data):
            curr_node = self.root
            while not curr_node.is_leaf():
                if instance[curr_node.feature] <= curr_node.test:
                    curr_node = curr_node.left_child
                else:
                    curr_node = curr_node.right_child
            predictions.append(curr_node.get_label())
        return predictions


def best_split(data, cost_function):
    split = {'cost': np.inf, 'feature': None, 'test': None, 'data': {'left': None, 'right': None}}

    for feature_index in range(data.shape[1] - 1):
        feature_vector = data.values[:, feature_index]
        feature_tests = np.unique(feature_vector)

        for test in feature_tests:
            data_left = data.loc[data[data.columns[feature_index]] <= test]
            data_right = data.loc[data[data.columns[feature_index]] > test]

            if data_left.shape[0] >= 0 and data_right.shape[0] >= 0:
                cost_left = cost_function(data_left.iloc[:, -1])
                cost_right = cost_function(data_right.iloc[:, -1])
                cost_weighted = ((data_left.shape[0] * cost_left) + (data_right.shape[0] * cost_right)) / data.shape[0]

                if cost_weighted < split['cost']:
                    split = {'cost': cost_weighted, 'feature': feature_index, 'test': test,
                             'data': {'left': data_left, 'right': data_right}}
    return split
