import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter


###Distance Metric : L2 ###
class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
    
    """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k):
        """
    STUDENT CODE
    """
        dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):
        """
        STUDENT CODE
        """

        dists = pairwise_distances(X, self.X_train, metric='euclidean')
        return dists

    def predict_labels(self, dists, k):
        """
        STUDENT CODE
        """
        def find_most_frequent(row_labels):
            label_counter = Counter(row_labels.tolist())
            return label_counter.most_common(1)[0][0]

        y_indices = np.argsort(dists, axis=1)[:, :k]
        labels = self.y_train[y_indices]
        y_pred = np.apply_along_axis(find_most_frequent, axis=1, arr=labels)
        return y_pred
