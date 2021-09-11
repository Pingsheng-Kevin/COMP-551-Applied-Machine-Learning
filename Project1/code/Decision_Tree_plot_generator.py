import numpy as np
import pandas as pds
import argparse
import warnings
import matplotlib.pyplot as plt
np.random.seed(5)


# cost functions available
def cost_function_misclassification_rate(labels):
    # labels = labels.astype(int)
    class_prob_arr = np.bincount(labels) / len(labels)
    return 1 - np.max(class_prob_arr)


def cost_function_entropy(labels):
    # labels = labels.astype(int)
    class_prob_arr = np.bincount(labels) / len(labels)
    class_prob_arr = class_prob_arr[class_prob_arr != 0]
    return np.sum(class_prob_arr * np.log2(1./class_prob_arr))


def cost_function_gini_index(labels):
    # labels = labels.astype(int)
    class_prob_arr = np.bincount(labels) / len(labels)
    return 1 - np.sum(np.square(class_prob_arr))


# the cost function by default is gini_index
def greedy_heuristic_tests_and_split(node, cost_function, switch):
    optimal_cost_value, optimal_feature, optimal_test_value, this_node_costs = np.inf, None, None, None
    (num_of_instances, num_of_features) = node.data.shape
    # print(node.data[node.data_instances_indices])
    data_sorted_by_features = np.sort(node.data[node.data_instances_indices], axis=0)

    # print(data_sorted_by_features[1:] + data_sorted_by_features[:-1])
    all_possible_tests_values = (data_sorted_by_features[1:] + data_sorted_by_features[:-1])/2.
    # print(all_possible_tests_values)
    for feature in range(0, num_of_features):
        # this is a vector
        # print(node.data_instances_indices)
        data_feature_value = node.data[node.data_instances_indices, feature]
        # print(data_feature_value)
        for t in all_possible_tests_values[:, feature]:
            left_child_instances_indices = node.data_instances_indices[data_feature_value <= t]
            right_child_instances_indices = node.data_instances_indices[data_feature_value > t]

            if len(left_child_instances_indices) == 0 or len(right_child_instances_indices) == 0:
                continue

            left_child_costs = cost_function(node.labels[left_child_instances_indices])
            right_child_costs = cost_function(node.labels[right_child_instances_indices])
            this_node_costs = cost_function(node.labels[node.data_instances_indices])

            cost_of_split = (left_child_instances_indices.shape[0] * left_child_costs +
                             right_child_instances_indices.shape[0] * right_child_costs) / num_of_instances
            # stop splitting if the switch is true and the change of cost is less than 10%
            if ((this_node_costs - cost_of_split)/(1. * this_node_costs) < 0.15) and switch:
                continue
            if cost_of_split < optimal_cost_value:
                optimal_cost_value = cost_of_split
                optimal_feature = feature
                optimal_test_value = t
    # print(optimal_cost_value, optimal_feature, optimal_test_value, this_node_costs)
    return optimal_cost_value, optimal_feature, optimal_test_value, this_node_costs


class DecisionTreeNode:
    # the cost function by default is gini_index
    def __init__(self, data_instances_indices, parent_node=None, criterion=None):
        self.data_instances_indices = data_instances_indices
        self.parent_node = parent_node
        self.criterion = criterion
        self.left_child, self.right_child = None, None
        self.feature_used_for_split, self.split_value = None, None
        if parent_node is not None:
            self.depth = parent_node.depth + 1
            self.num_of_classes = parent_node.num_of_classes
            self.labels = parent_node.labels
            self.data = parent_node.data
            class_prob_arr = np.bincount(self.labels[data_instances_indices], minlength=self.num_of_classes+1)
            self.class_prob_arr = class_prob_arr / np.sum(class_prob_arr)


class DecisionTreeClassifier:
    # the cost function by default is gini_index
    def __init__(self, num_of_classes=0, maximum_depth=100, criterion=cost_function_gini_index,
                 minimum_leaves=1, switch=False):
        self.num_of_classes = num_of_classes
        self.maximum_depth = maximum_depth
        self.criterion = criterion
        self.minimum_leaves = minimum_leaves
        self.root, self.data, self.labels = None, None, None
        self.switch = switch

    def fit(self, x_train_data, y_train_labels):
        self.data = x_train_data
        self.labels = y_train_labels
        if self.num_of_classes == 0:
            # here we assume each class has been encoded as 1, 2, 3, 4...
            self.num_of_classes = np.max(self.labels)
        self.root = DecisionTreeNode(np.arange(self.data.shape[0]), criterion=self.criterion)
        self.root.data = x_train_data
        self.root.labels = y_train_labels
        self.root.num_of_classes = self.num_of_classes
        self.root.depth = 0

        self._fit_decision_tree_(self.root)
        return self

    # recursive function
    def _fit_decision_tree_(self, node):
        if node.depth > self.maximum_depth:
            return
        cost, feature_used, test_value, cost_of_this_node = greedy_heuristic_tests_and_split(node, self.criterion,
                                                                                             self.switch)
        if np.isinf(cost):
            return
        # test is a boolean array
        test = (node.data[node.data_instances_indices, feature_used] <= test_value)
        node.feature_used_for_split = feature_used
        node.split_value = test_value

        left_child = DecisionTreeNode(node.data_instances_indices[test], parent_node=node, criterion=self.criterion)
        right_child = DecisionTreeNode(node.data_instances_indices[np.logical_not(test)], parent_node=node,
                                       criterion=self.criterion)

        self._fit_decision_tree_(left_child)
        self._fit_decision_tree_(right_child)

        node.left_child = left_child
        node.right_child = right_child

    def predict(self, instances):
        class_probability_matrix = np.zeros((instances.shape[0], self.num_of_classes + 1))
        # print(self.num_of_classes)
        # print(class_probability_matrix)
        for instance_index, instance_attributes in enumerate(instances):
            node_pointer = self.root
            while node_pointer.left_child is not None:
                if instance_attributes[node_pointer.feature_used_for_split] <= node_pointer.split_value:
                    node_pointer = node_pointer.left_child
                else:
                    node_pointer = node_pointer.right_child
            # print(node_pointer.class_prob_arr)
            class_probability_matrix[instance_index, :] = node_pointer.class_prob_arr
        # here we assume the index of the one has greatest prob is the class, starting from 1
        return np.argmax(class_probability_matrix, axis=1)

    # both are np.array
    def evaluate_acc(self, true_labels, predicted_labels):
        result = np.bincount(np.array(true_labels) == np.array(predicted_labels))
        if len(result) == 1:
            return 0.
        else:
            # print(result)
            return result[1] / np.sum(result)


def preprocess_hepatitis_data(filepath):
    # load and clean the hepatitis data
    df = pds.read_csv(filepath)
    df_clean = df[~df.eq('?').any(1)]
    return df_clean


def preprocess_cancer_data(filepath):
    # load and clean the cancer data
    df = pds.read_csv(filepath)
    df_clean = df[~df.eq('?').any(1)]
    df_clean = df_clean.drop('id', axis='columns')
    df_clean['Class'] = df_clean['Class'].__floordiv__(2)
    return df_clean


def run_experiment_cancer(dataframe):
    dataset = dataframe.to_numpy(dtype='float')
    inputs, labels = dataset[:, :-1], dataset[:, -1].astype(int)
    (num_of_instances, num_of_features), num_of_classes = inputs.shape, np.max(labels)
    sample_indices = np.random.permutation(num_of_instances)
    # 70% of data is used for training, the rest for testing
    cut_off = int(num_of_instances * 3/10)
    x_train, y_train = inputs[sample_indices[:cut_off]], labels[sample_indices[:cut_off]]
    x_test, y_test = inputs[sample_indices[cut_off:]], labels[sample_indices[cut_off:]]

    DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_gini_index,
                                               maximum_depth=100, switch=False)
    DecisionTreeModel.fit(x_train, y_train)
    y_prediction = DecisionTreeModel.predict(x_test)
    accuracy = DecisionTreeModel.evaluate_acc(y_test, y_prediction)
    print(f'The accuracy is: {accuracy*100:.2f}%')

    correct_predictions = y_prediction == y_test
    incorrect_predictions = np.logical_not(correct_predictions)
    for i in range(0, num_of_features):
        for j in range(i, num_of_features):
            plt.xlabel(f'feature: {i}')
            plt.ylabel(f'feature: {j}')
            xiv = np.linspace(np.min(inputs[:, i]), np.max(inputs[:, i]), 200)
            xjv = np.linspace(np.min(inputs[:, j]), np.max(inputs[:, j]), 200)
            xi, xj = np.meshgrid(xiv, xjv)
            x_all = np.vstack((xi.ravel(), xj.ravel())).T
            x_final = np.zeros((x_all.shape[0], num_of_features))
            x_final[:, [i, j]] = x_all[:, [0, 1]]
            # print(x_final)

            y_train_prob = np.zeros((y_train.shape[0], num_of_classes+1))
            # print(y_train_prob)
            y_train_prob[np.arange(y_train.shape[0]), y_train] = 1

            y_test_prob = np.zeros((y_test.shape[0], num_of_classes+1))
            y_test_prob[np.arange(y_test.shape[0]), y_test] = 1
            y_prob_all = np.zeros((x_final.shape[0], num_of_classes+1))
            y_prob_all[np.arange(x_final.shape[0]), DecisionTreeModel.predict(x_final)] = 1

            plt.scatter(x_train[:, i], x_train[:, j], c=y_train_prob, marker='o', alpha=0.5, label='training')
            plt.scatter(x_test[:, i], x_test[:, j], c=y_test_prob, marker='x', alpha=0.5, label='testing')
            plt.scatter(x_final[:, i], x_final[:, j], c=y_prob_all, marker='.', alpha=.01)
            """
            plt.scatter(x_train[:, i], x_train[:, j], c=y_train, marker='o', alpha=.1)
            plt.scatter(x_test[correct_predictions, i], x_test[correct_predictions, j], marker='+',
                        c=y_prediction[correct_predictions], alpha=.2, label='correct')
            plt.scatter(x_test[incorrect_predictions, i], x_test[incorrect_predictions, j], marker='x',
                        c=y_test[incorrect_predictions], alpha=.3, label='incorrect')
            plt.legend()
            """
            plt.show()



def run_experiment_hepatitis(dataframe):
    dataset = dataframe.to_numpy(dtype='float')
    inputs, labels = dataset[:, 1:], dataset[:, 0].astype(int)
    (num_of_instances, num_of_features), num_of_classes = inputs.shape, np.max(labels)
    sample_indices = np.random.permutation(num_of_instances)
    # 70% of data is used for training, the rest for testing
    cut_off = int(num_of_instances * 8 / 10)
    x_train, y_train = inputs[sample_indices[:cut_off]], labels[sample_indices[:cut_off]]
    x_test, y_test = inputs[sample_indices[cut_off:]], labels[sample_indices[cut_off:]]

    DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_gini_index,
                                               maximum_depth=100, switch=False, minimum_leaves=1)
    DecisionTreeModel.fit(x_train, y_train)
    y_prediction = DecisionTreeModel.predict(x_test)
    accuracy = DecisionTreeModel.evaluate_acc(y_test, y_prediction)
    print(f'The accuracy is: {accuracy*100.0:.2f}%')

    # plot
    """
    correct_predictions = y_prediction == y_test
    incorrect_predictions = np.logical_not(correct_predictions)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='o', alpha=.2)
    plt.scatter(x_test[correct_predictions, 0], x_test[correct_predictions, 1], marker='+',
                c=y_prediction[correct_predictions], label='correct')
    plt.scatter(x_test[incorrect_predictions, 0], x_test[incorrect_predictions, 1], marker='x',
                c=y_prediction[correct_predictions], label='incorrect')
    plt.legend()
    """
    correct_predictions = y_prediction == y_test
    incorrect_predictions = np.logical_not(correct_predictions)
    for i in range(10, num_of_features):
        for j in range(i, num_of_features):
            plt.xlabel(f'feature: {i}')
            plt.ylabel(f'feature: {j}')
            xiv = np.linspace(np.min(inputs[:, i]), np.max(inputs[:, i]), 200)
            xjv = np.linspace(np.min(inputs[:, j]), np.max(inputs[:, j]), 200)
            xi, xj = np.meshgrid(xiv, xjv)
            x_all = np.vstack((xi.ravel(), xj.ravel())).T
            x_final = np.ones((x_all.shape[0], num_of_features)) * 1
            x_final[:, [i, j]] = x_all[:, [0, 1]]
            # print(x_final)

            y_train_prob = np.zeros((y_train.shape[0], num_of_classes + 1))
            # print(y_train_prob)
            y_train_prob[np.arange(y_train.shape[0]), y_train] = 1

            y_test_prob = np.zeros((y_test.shape[0], num_of_classes + 1))
            y_test_prob[np.arange(y_test.shape[0]), y_test] = 1
            y_prob_all = np.zeros((x_final.shape[0], num_of_classes + 1))
            y_prob_all[np.arange(x_final.shape[0]), DecisionTreeModel.predict(x_final)] = 1

            plt.scatter(x_train[:, i], x_train[:, j], c=y_train_prob, marker='o', alpha=0.3)
            plt.scatter(x_test[:, i], x_test[:, j], c=y_test_prob, marker='x', alpha=0.3)
            plt.scatter(x_final[:, i], x_final[:, j], c=y_prob_all, marker='.', alpha=.01)
            """
            plt.scatter(x_train[:, i], x_train[:, j], c=y_train, marker='o', alpha=.1)
            plt.scatter(x_test[correct_predictions, i], x_test[correct_predictions, j], marker='+',
                        c=y_prediction[correct_predictions], alpha=.2, label='correct')
            plt.scatter(x_test[incorrect_predictions, i], x_test[incorrect_predictions, j], marker='x',
                        c=y_test[incorrect_predictions], alpha=.3, label='incorrect')
            plt.legend()
            """
            # plt.legend()
            plt.show()
    plt.show()


def main():
    warnings.filterwarnings(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("-c", "--cancer", action='store_true', help="this flag means the input is the cancer file")
    parser.add_argument("-H", "--hepatitis", action='store_true')
    args = parser.parse_args()
    if args.cancer:
        df = preprocess_cancer_data(args.data_path)
        # print(df)
        run_experiment_cancer(df)
    elif args.hepatitis:
        df = preprocess_hepatitis_data(args.data_path)
        # print(df)
        run_experiment_hepatitis(df)
    else:
        print('This file is unknown, therefore unable to process.')
        quit(0)


if __name__ == '__main__':
    main()
