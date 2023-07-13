import numpy as np
class KMeans:
    
    def __init__(self, n_clusters, dist_fn, mean_fn, random_seed):
        self.K = n_clusters
        self.dist_fn = dist_fn
        self.mean_fn = mean_fn
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), self.K, replace=False)]
        self.intial_centroids = self.centroids
        self.prev_label,  self.labels = None, np.zeros(len(X))
        while not np.all(self.labels == self.prev_label):
            self.prev_label = self.labels
            self.labels = self.predict(X)
            self.update_centroid(X)
        return self
        
    def predict(self, X):
        return np.apply_along_axis(self.compute_label, 1, X)

    def compute_label(self, x):
        dist = []
        for center in self.centroids:
            dist.append(self.dist_fn(center, x))
        return np.argmin(np.array(dist))
    
    def get_average_distance(self, X, result):
        dists = [0 for i in range(self.K)]
        count = [0 for i in range(self.K)]
        for i, x in enumerate(X):
            dists[result[i]] += self.dist_fn(self.centroids[result[i]], x)
            count[result[i]] += 1
        for i in range(self.K):
            dists[i] /= count[i]
        return dists
    
    def compute_dist(self, x):
        label = self.compute_label(x)
        return self.dist_fn(self.centroids[label], x)
    def update_centroid(self, X):
        self.centroids = np.array([self.mean_fn(X[self.labels == k])  for k in range(self.K)])

