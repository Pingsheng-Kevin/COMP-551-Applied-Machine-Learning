import numpy as np
import Dist


def aux_insort(ar, e, K, key):
    """
    insort ar and pop exceeding element on the right
    :param ar: the input array
    :param e: new element needed to insert
    :param K: max num of element in ar
    :param key: comparator
    :return: void
    """
    lo = 0
    hi = len(ar) - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        if key(e) < key(ar[mid]):
            hi = mid - 1
        else:
            lo = mid + 1

    if lo > len(ar) - 1:
        ar.append(e)
    else:
        ar.insert(lo, e)

    if len(ar) > K:
        ar.pop()


class Node(object):
    def __init__(self, dimension, Dataset_X, dist_fn):
        self.left = None
        self.right = None
        self.index = None
        self.D = dimension
        self.isLeaf = False
        self.dist_fn = dist_fn
        self.Dataset_X = Dataset_X

    def _get_dist(self, xi):
        return self.dist_fn(xi, self.Dataset_X[self.index])

    def _get_hyperplane_dist(self, xi):
        return abs(xi[self.D] - self.Dataset_X[self.index][self.D])

    def nnsearch(self, xi, K, nearest):
        """
        Nearest from smallest to largest
        """
        cur_dist = self._get_dist(xi)

        cap = len(nearest)

        if not self:
            return

        if self.isLeaf:

            if cap < K or cur_dist < nearest[-1][-1]:
                aux_insort(nearest, (self.index, cur_dist), K, lambda tup: tup[-1])

        else:

            if xi[self.D] <= self.Dataset_X[self.index][self.D]:
                if self.left is not None:
                    self.left.nnsearch(xi, K, nearest)
            else:
                if self.right is not None:
                    self.right.nnsearch(xi, K, nearest)

            hyper_dist = self._get_hyperplane_dist(xi)

            cap = len(nearest)

            if cap < K or hyper_dist < nearest[-1][-1]:

                if cur_dist < nearest[-1][-1]:
                    aux_insort(nearest, (self.index, cur_dist), K, lambda tup: tup[-1])

                if xi[self.D] <= self.Dataset_X[self.index][self.D]:
                    if self.right is not None:
                        self.right.nnsearch(xi, K, nearest)
                else:
                    if self.left is not None:
                        self.left.nnsearch(xi, K, nearest)


class KDTree(object):
    def __init__(self, k, Dataset_X, dist_fn=Dist.euclidean):
        self.root = Node(0, Dataset_X, dist_fn)
        self.Dataset_X = Dataset_X
        self.k = k
        self.dist_fn = dist_fn
        self._build()

    def _split_data(self, indexes, dimension, start, end):
        """Split the indexes into left, mid, right
        according to the specified dimension
        """
        mid = start + (end - start) // 2 + ((end - start) % 2 > 0)
        indexes[start:end] = sorted(indexes[start:end], key=lambda index: self.Dataset_X[index][dimension])
        return mid, indexes[mid]

    def _build(self):
        indexes = np.arange(self.Dataset_X.shape[0])
        stack = [(self.root, 0, len(indexes) - 1)]

        while stack:
            # each current node is responsible to initialize its left and right, its index, and children's dimension.
            # initialize root node with dimension 0
            cur_node, cur_start, cur_end = stack.pop()
            if cur_start == cur_end:
                cur_node.isLeaf = True
                cur_node.left = None
                cur_node.right = None
                cur_node.index = indexes[cur_start]
                continue

            mid, mid_index = self._split_data(indexes, cur_node.D, cur_start, cur_end)
            cur_node.index = mid_index

            left_start = cur_start
            left_end = mid - 1
            right_start = mid + 1
            right_end = cur_end
            if right_start <= right_end:
                cur_node.right = Node((cur_node.D + 1) % self.k, self.Dataset_X, self.dist_fn)
                stack.append((cur_node.right, right_start, right_end))

            cur_node.left = Node((cur_node.D + 1) % self.k, self.Dataset_X, self.dist_fn)
            stack.append((cur_node.left, left_start, left_end))

    def nearest_neighbour_search(self, xi, K):
        nearest = [(self.root.index, float('inf'))]
        self.root.nnsearch(xi, K, nearest)
        return nearest
