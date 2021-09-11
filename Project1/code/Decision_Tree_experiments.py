import numpy as np
import pandas as pds
import argparse
import warnings
np.random.seed(12349)


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
def greedy_heuristic_tests_and_split(node, cost_function, switch, stop):
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
            if ((this_node_costs - cost_of_split)/(1. * this_node_costs) < stop) and switch:
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
        self.is_leaf = False
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
                 minimum_leaves=1, switch=False, stop=0.1):
        self.num_of_classes = num_of_classes
        self.maximum_depth = maximum_depth
        self.criterion = criterion
        self.minimum_leaves = minimum_leaves
        self.root, self.data, self.labels = None, None, None
        self.switch = switch
        self.stop = stop

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
        if node.depth == self.maximum_depth or len(node.data_instances_indices) <= self.minimum_leaves:
            return
        cost, feature_used, test_value, cost_of_this_node = greedy_heuristic_tests_and_split(node, self.criterion,
                                                                                             self.switch, self.stop)
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

    def post_pruning(self, valid_data=None, valid_labels=None):
        leaves_stack = self.get_all_leaves()
        leaves_stack_no_subtree = []
        for leaf in leaves_stack:
            leaf.is_leaf = True

        for leaf in leaves_stack:
            if leaf.parent_node.left_child.is_leaf and leaf.parent_node.right_child.is_leaf \
                    and leaf not in leaves_stack_no_subtree:
                leaves_stack_no_subtree.append(leaf)
        while len(leaves_stack) > 1 and len(leaves_stack_no_subtree) > 1:

            # print(leaves_stack_no_subtree)

            leaf = leaves_stack_no_subtree.pop()
            another_leaf = None
            for l in leaves_stack_no_subtree:
                if l.parent_node is leaf.parent_node:
                    another_leaf = l
                    leaves_stack_no_subtree.remove(another_leaf)
            if another_leaf is None:
                print('something wrong happens')

            leaf_parent = leaf.parent_node
            # stop if there is only root
            if leaf_parent is self.root:
                return
            accuracy_before = self.evaluate_acc(valid_labels, self.predict(valid_data))
            # print(f'The accuracy on valid set before pruning: {100 * accuracy_before:.2f}%')
            # try to replace the 2 leaves by their parent
            store_left = leaf_parent.left_child
            store_right = leaf_parent.right_child
            leaf_parent.left_child = None
            leaf_parent.right_child = None
            leaf_parent.class_prob_arr = (len(store_left.data_instances_indices) * store_left.class_prob_arr
                                          +
                                          len(store_right.data_instances_indices) * store_right.class_prob_arr) /\
                                         (len(store_left.data_instances_indices) +
                                             len(store_right.data_instances_indices))
            accuracy_after = self.evaluate_acc(valid_labels, self.predict(valid_data))
            # print(f'The accuracy on valid set after this pruning operation would be: {100 * accuracy_after:.2f}%')
            # no need for pruning
            if accuracy_before >= accuracy_after:
                leaf_parent.left_child = store_left
                leaf_parent.right_child = store_right
                leaf_parent.class_prob_arr = None
                leaves_stack.remove(leaf)
                leaves_stack.remove(another_leaf)
                # print('Pruning operation reversed due to damage on accuracy or no effect.')
            else:
                leaves_stack.remove(leaf)
                leaves_stack.remove(another_leaf)
                store_left.parent_node = None
                store_right.parent_node = None
                leaf_parent.is_leaf = True
                leaves_stack.append(leaf_parent)
                leaf_parent.feature_used_for_split, leaf_parent.split_value = None, None
                # print('Pruning operation is successful.')
                for leaf in leaves_stack:
                    if leaf.parent_node.left_child.is_leaf and leaf.parent_node.right_child.is_leaf \
                            and leaf not in leaves_stack_no_subtree:
                        leaves_stack_no_subtree.append(leaf)

    def get_all_leaves(self):
        stack_1 = []
        stack_2 = []
        stack_1.append(self.root)
        while len(stack_1) != 0:
            current = stack_1.pop()
            if current.left_child:
                stack_1.append(current.left_child)
            if current.right_child:
                stack_1.append(current.right_child)
            elif not current.left_child and not current.right_child:
                stack_2.append(current)
        return stack_2


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


def run_experiment_cancer(dataframe, post_pruning=False):
    dataset = dataframe.to_numpy(dtype='float')
    inputs, labels = dataset[:, :-1], dataset[:, -1].astype(int)
    (num_of_instances, num_of_features), num_of_classes = inputs.shape, np.max(labels)
    acc_list_gini = []
    acc_list_entropy = []
    acc_list_mis = []
    for run in range(0, 20):

        sample_indices = np.random.permutation(num_of_instances)
        # 80% of data is used for training/validation, the rest for testing
        # performing cross-validation by modifying this part manually
        cut_off = int(num_of_instances * 8 / 10)
        train_off = int(cut_off * 4 / 5)
        if not post_pruning:
            x_train, y_train = inputs[sample_indices[:cut_off]], labels[sample_indices[:cut_off]]
            x_test, y_test = inputs[sample_indices[cut_off:]], labels[sample_indices[cut_off:]]
        elif post_pruning:
            x_train, y_train = inputs[sample_indices[:train_off]], labels[sample_indices[:train_off]]
            x_valid, y_valid = inputs[sample_indices[train_off:cut_off]], labels[sample_indices[train_off:cut_off]]
            x_test, y_test = inputs[sample_indices[cut_off:]], labels[sample_indices[cut_off:]]

        def run_experiment(DecisionTreeModel):
            print('The hyper-parameters for this model: ')
            print('Max_depth: ', DecisionTreeModel.maximum_depth)
            print('Minimum number of data in leaves nodes: ', DecisionTreeModel.minimum_leaves)
            print('The cost function: ', DecisionTreeModel.criterion.__name__)
            print('Stop splitting if no substantial cost reduction was made (T/F)?: ', DecisionTreeModel.switch)
            print('Post-pruning enabled (T/F)?: ', post_pruning)
            print('-----------------------------------------------')
            DecisionTreeModel.fit(x_train, y_train)
            y_prediction = DecisionTreeModel.predict(x_test)
            accuracy = DecisionTreeModel.evaluate_acc(y_test, y_prediction)
            print(f'The accuracy is: {accuracy * 100.0:.2f}%')
            if post_pruning:
                DecisionTreeModel.post_pruning(valid_data=x_valid, valid_labels=y_valid)
                accuracy = DecisionTreeModel.evaluate_acc(y_test, y_prediction)
                print(f'The accuracy after pruning operation is: {accuracy * 100.0:.2f}%')
            print('-----------------------------------------------')
            return accuracy*100

        for j in range(1, 2):
            for i in range(3, 4):

                DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_gini_index,
                                                           maximum_depth=i, switch=False, minimum_leaves=5 * j)
                run_experiment(DecisionTreeModel)

                acc = run_experiment(DecisionTreeModel)
                acc_list_gini.append(acc)

                DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_entropy,
                                                           maximum_depth=i, switch=False, minimum_leaves=5 * j)
                run_experiment(DecisionTreeModel)

                acc = run_experiment(DecisionTreeModel)
                acc_list_entropy.append(acc)

                DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_misclassification_rate,
                                                           maximum_depth=i, switch=False, minimum_leaves=5 * j)
                acc = run_experiment(DecisionTreeModel)
                acc_list_mis.append(acc)
    gini_np = np.array(acc_list_gini)
    entropy_np = np.array(acc_list_entropy)
    mis_np = np.array(acc_list_mis)
    print('Average(GINI): ', gini_np.sum()/len(gini_np), 'std = ', gini_np.std())
    print('Average(Entropy): ', entropy_np.sum()/len(entropy_np), 'std = ', entropy_np.std())
    print('Average(Mis): ', mis_np.sum()/len(mis_np), 'std = ', mis_np.std())

    # The optimal solution
    DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_misclassification_rate,
                                               maximum_depth=3, switch=False, minimum_leaves=5)
    run_experiment(DecisionTreeModel)

    for i in range(0, 6):

        DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_gini_index,
                                                   maximum_depth=100, switch=True, minimum_leaves=1, stop=0.1+(i*0.05))
        run_experiment(DecisionTreeModel)
        print(f'The split has stop when reduction rate is lower than: {DecisionTreeModel.stop*100:.2f}%')

        DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_entropy,
                                                   maximum_depth=100, switch=True, minimum_leaves=1,
                                                   stop=0.1 + (i * 0.05))
        run_experiment(DecisionTreeModel)
        print(f'The split has stop when reduction rate is lower than: {DecisionTreeModel.stop * 100:.2f}%')

        DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_misclassification_rate,
                                                   maximum_depth=100, switch=True, minimum_leaves=1,
                                                   stop=0.1 + (i * 0.05))
        run_experiment(DecisionTreeModel)
        print(f'The split has been stopped when reduction rate is lower than: {DecisionTreeModel.stop * 100:.2f}%')


def run_experiment_hepatitis(dataframe, post_pruning=False):

    dataset = dataframe.to_numpy(dtype='float')
    inputs, labels = dataset[:, 1:], dataset[:, 0].astype(int)
    (num_of_instances, num_of_features), num_of_classes = inputs.shape, np.max(labels)
    acc_list_gini = []
    acc_list_entropy = []
    acc_list_mis = []
    for run in range(0, 20):

        sample_indices = np.random.permutation(num_of_instances)
        # 80% of data is used for training/validation, the rest for testing
        # performing cross-validation by modifying this part manually
        cut_off = int(num_of_instances * 8 / 10)
        train_off = int(cut_off * 4 / 5)
        if not post_pruning:
            x_train, y_train = inputs[sample_indices[:cut_off]], labels[sample_indices[:cut_off]]
            x_test, y_test = inputs[sample_indices[cut_off:]], labels[sample_indices[cut_off:]]
        elif post_pruning:
            x_train, y_train = inputs[sample_indices[:train_off]], labels[sample_indices[:train_off]]
            x_valid, y_valid = inputs[sample_indices[train_off:cut_off]], labels[sample_indices[train_off:cut_off]]
            x_test, y_test = inputs[sample_indices[cut_off:]], labels[sample_indices[cut_off:]]

        def run_experiment(DecisionTreeModel):
            print('The hyper-parameters for this model: ')
            print('Max_depth: ', DecisionTreeModel.maximum_depth)
            print('Minimum number of data in leaves nodes: ', DecisionTreeModel.minimum_leaves)
            print('The cost function: ', DecisionTreeModel.criterion.__name__)
            print('Stop splitting if no substantial cost reduction was made (T/F)?: ', DecisionTreeModel.switch)
            print('Post-pruning enabled (T/F)?: ', post_pruning)
            print('-----------------------------------------------')
            DecisionTreeModel.fit(x_train, y_train)
            y_prediction = DecisionTreeModel.predict(x_test)
            accuracy = DecisionTreeModel.evaluate_acc(y_test, y_prediction)
            print(f'The accuracy is: {accuracy * 100.0:.2f}%')
            if post_pruning:
                DecisionTreeModel.post_pruning(valid_data=x_valid, valid_labels=y_valid)
                accuracy = DecisionTreeModel.evaluate_acc(y_test, y_prediction)
                print(f'The accuracy after pruning operation is: {accuracy * 100.0:.2f}%')
            print('-----------------------------------------------')
            return accuracy * 100

        for j in range(1, 2):
            for i in range(3, 4):
                DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_gini_index,
                                                           maximum_depth=i, switch=False, minimum_leaves=5 * j)
                run_experiment(DecisionTreeModel)

                acc = run_experiment(DecisionTreeModel)
                acc_list_gini.append(acc)

                DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_entropy,
                                                           maximum_depth=i, switch=False, minimum_leaves=5 * j)
                run_experiment(DecisionTreeModel)

                acc = run_experiment(DecisionTreeModel)
                acc_list_entropy.append(acc)

                DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_misclassification_rate,
                                                           maximum_depth=i, switch=False, minimum_leaves=5 * j)
                acc = run_experiment(DecisionTreeModel)
                acc_list_mis.append(acc)
    gini_np = np.array(acc_list_gini)
    entropy_np = np.array(acc_list_entropy)
    mis_np = np.array(acc_list_mis)
    print('Average(GINI): ', gini_np.sum() / len(gini_np), 'std = ', gini_np.std())
    print('Average(Entropy): ', entropy_np.sum() / len(entropy_np), 'std = ', entropy_np.std())
    print('Average(Mis): ', mis_np.sum() / len(mis_np), 'std = ', mis_np.std())

    DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_misclassification_rate,
                                               maximum_depth=4, switch=False, minimum_leaves=5)
    run_experiment(DecisionTreeModel)

    for i in range(0, 5):

        DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_gini_index,
                                                   maximum_depth=100, switch=True, minimum_leaves=1, stop=0.1+(i*0.05))
        run_experiment(DecisionTreeModel)
        print(f'The split has stop when reduction rate is lower than: {DecisionTreeModel.stop*100:.2f}%')

        DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_entropy,
                                                   maximum_depth=100, switch=True, minimum_leaves=1,
                                                   stop=0.1 + (i * 0.05))
        run_experiment(DecisionTreeModel)
        print(f'The split has stop when reduction rate is lower than: {DecisionTreeModel.stop * 100:.2f}%')

        DecisionTreeModel = DecisionTreeClassifier(criterion=cost_function_misclassification_rate,
                                                   maximum_depth=100, switch=True, minimum_leaves=1,
                                                   stop=0.1 + (i * 0.05))
        run_experiment(DecisionTreeModel)
        print(f'The split has been stopped when reduction rate is lower than: {DecisionTreeModel.stop * 100:.2f}%')


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
