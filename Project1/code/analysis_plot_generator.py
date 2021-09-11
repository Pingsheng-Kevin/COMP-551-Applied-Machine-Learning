import numpy as np

import matplotlib.pyplot as plt


def plot_accuracy_stop():
    x_values = [5*i for i in range(0, 8)]
    y_values_hepatitis = [96.35, 96.35, 97.81, 97.81, 97.81, 97.81, 97.81 , 97.08]

    plt.plot(x_values, y_values_hepatitis, marker='*', color='red', linewidth=2)
    plt.xlabel('Cost Reduction Threshold (%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Cost Reduction Threshold')
    # plt.legend()
    plt.show()


def plot_accuracy_min_leaves():
    x_values = [2*i for i in range(1, 16)]
    y_values_cancer = [92.7007299270073, 92.7007299270073, 92.7007299270073,
                       92.7007299270073, 94.16058394160584, 93.43065693430657,
                       93.43065693430657, 93.43065693430657, 93.43065693430657,
                       93.43065693430657, 93.43065693430657, 92.7007299270073,
                       92.7007299270073, 92.7007299270073, 92.7007299270073]
    plt.plot(x_values, y_values_cancer, marker='s', markerfacecolor='black', color='orange', linewidth=2,
             label='breast cancer')
    plt.xlabel('Minimum number of data in each leaf')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Minimum number of data in each leaf')
    plt.legend()
    plt.show()


def plot_accuracy_max_depth():
    x_values = [i for i in range(1, 11)]
    y_values_hepatitis = [80, 88, 88, 76, 72, 72, 72, 72, 72, 72]

    plt.plot(x_values, y_values_hepatitis, marker='x', markerfacecolor='blue', color='olive', linewidth=2,
             label='hepatitis')
    plt.xlabel('Maximum depth values')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Maximum depth values ')
    plt.legend()
    plt.show()

def plot_accuracy_K():
    x_values = [i for i in range(1, 11)]
    y_values_cancer = [94.73, 94.73, 95.07, 95.17, 95.61, 95.61, 95.58, 95.58, 95.17, 95.17]

    y_values_hepatitis = [74.07, 74.07, 77.77, 77.77, 85.18, 85.18, 88.88, 85.18, 92.59, 88.88]

    plt.plot(x_values, y_values_cancer, marker='o', markerfacecolor='blue', color='skyblue', linewidth=2,
             label='breast cancer')
    """
    plt.plot(x_values, y_values_hepatitis, marker='x', color='olive', linewidth=2, linestyle='dashed',
             label='hepatitis')
    """
    plt.xlabel(' K values')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. K values ')
    plt.legend()
    plt.show()


def main():
    plot_accuracy_K()
    plot_accuracy_max_depth()
    plot_accuracy_min_leaves()
    plot_accuracy_stop()


if __name__ == '__main__':
    main()
