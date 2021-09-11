import numpy as np
import matplotlib.pyplot as plt
import Dist
from KNN import KNN
import Datasets
# np.random.seed(8090)

if __name__ == '__main__':
    dataset_can = Datasets.load_cancer()
    dataset_hepa = Datasets.load_hepatitis()

    for dataset in [dataset_can, dataset_hepa]:
        x, y = np.copy(dataset['data']), np.copy(dataset['target'])
        (N, D) = x.shape
        num_train = (2 * N) // 3
        print(f'instances (N) \t {N}')
        print(f"features (D) \t {D}")

        # normalized using standard scaling
        means = np.average(x, axis=0)
        devis = np.zeros_like(means)
        for i in range(x.shape[-1]):
            devis[i] = np.sqrt(np.sum((x[:, i] - means[i]) ** 2) / N)
            x[:, i] = (x[:, i] - means[i]) / devis[i]

        acc_mant_list = []
        acc_eucl_list = []
        acc_uniform_list = []
        acc_distance_list = []

        for i in range(0, 20):
            inds = np.random.permutation(N)

            # experiments with different combinations, now only optimal K values are kept

            for K in (5, 10):
                for voting in ['uniform', 'distance']:
                    for dist_fn in [Dist.manhattan, Dist.euclidean]:
                        x_train, y_train = x[inds[:num_train]], y[inds[:num_train]]
                        x_test, y_test = x[inds[num_train:]], y[inds[num_train:]]

                        model_fit = KNN(K=K, algorithm='brute', voting=voting, dist_fn=dist_fn).fit(x_train, y_train)
                        x_test_pred = model_fit.predict(x_test)
                        score_test = KNN.evaluate_acc(x_test_pred, y_test)

                        x_train_pred = model_fit.predict(x_train)
                        score_train = KNN.evaluate_acc(x_train_pred, y_train)

                        print(f'K: {K} voting: {voting}')
                        print(f'Distance Function: {"Euclidean" if dist_fn is Dist.euclidean else "Manhattan"}')
                        print(f'testing accuracy : {score_test * 100}%   training accuracy : {score_train * 100}')

                        if voting == 'distance' and dist_fn is Dist.manhattan:
                            acc_mant_list.append(score_test * 100)
                            acc_distance_list.append(score_test * 100)
                            # acc_train_list.append(score_train * 100)
                        elif voting == 'distance' and dist_fn is Dist.euclidean:
                            acc_eucl_list.append(score_test * 100)
                        if voting == 'uniform' and dist_fn is Dist.manhattan:
                            acc_uniform_list.append(score_test * 100)

        acc_mant_np = np.array(acc_mant_list)
        acc_eucl_np = np.array(acc_eucl_list)
        acc_distance_np = np.array(acc_distance_list)
        acc_uniform_np = np.array(acc_uniform_list)
        print('Average of accuracy of Manhattan:', acc_mant_np.sum()/len(acc_mant_np), 'std = ', acc_mant_np.std())
        print('Average of accuracy of Euclidean:', acc_eucl_np.sum() / len(acc_eucl_np), 'std = ',
              acc_eucl_np.std())
        print('Average of accuracy of Uniform: ', acc_uniform_np.sum()/len(acc_uniform_list), 'std = ',
              acc_uniform_np.std())
        print('Average of accuracy of Distance: ', acc_distance_np.sum()/len(acc_distance_list), 'std = ',
              acc_distance_np.std())



    # Plot one picture of Hepa Dataset with decision boundary
    x, y = np.copy(dataset_can['data']), np.copy(dataset_can['target'])
    (N, D) = x.shape
    means = np.average(x, axis=0)
    devis = np.zeros_like(means)
    for i in range(x.shape[-1]):
        devis[i] = np.sqrt(np.sum((x[:, i] - means[i]) ** 2) / N)
        x[:, i] = (x[:, i] - means[i]) / devis[i]

    f1, f2 = dataset_can['features'][2], dataset_can['features'][3]
    x0v = np.linspace(np.min(x[:, 2]), np.max(x[:, 2]), 100)
    x1v = np.linspace(np.min(x[:, 3]), np.max(x[:, 3]), 100)
    x0, x1 = np.meshgrid(x0v, x1v)
    x01 = np.stack((x0.ravel(), x1.ravel()), axis=-1)

    x_all = np.zeros((10000, D))
    x_all[:, 2] = x01[:, 0]
    x_all[:, 3] = x01[:, -1]
    gen = (i for i in range(0, D) if i != 2 and i != 3)
    for i in gen:
        x_all[:, i] = 0

    model_fit = KNN(K=K, algorithm='brute', voting='uniform', dist_fn=Dist.euclidean).fit(x, y)
    x_all_pred = model_fit.predict(x_all)

    plt.scatter(x_all[:, 2], x_all[:, 3], c=x_all_pred, marker='.', alpha=.1)

    plt.ylabel(f2)
    plt.xlabel(f1)
    plt.show()
