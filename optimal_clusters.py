from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.mixture import GaussianMixture

from scipy.optimize import minimize

import numpy as np

import time

class Data_Generator():
    def __init__(self, num_clusters, data_points):

        self.num_clusters = num_clusters
        self.data_points = data_points
        self.centers = []
        self.data = []

        for i in range(self.num_clusters):
            cluster_center = self.pickCenters(self.centers,self.num_clusters)
            new_data = self.generateData(cluster_center, int(self.data_points/self.num_clusters))
            self.centers.append(cluster_center)
            self.data.append(new_data)

        self.data = np.vstack(self.data)

    def generateData(self, cluster_center, data_points):
        
        mean = [cluster_center[0],cluster_center[1]]
        cov = [[1,0], [0,1]]
        x = np.random.multivariate_normal(mean, cov, data_points)
        return x

    def pickCenters(self,centers,num_clusters):
        cluster_center = np.random.uniform(-50,50,2)
        return [cluster_center[0], cluster_center[1]]
        


def cluster_and_score(data, num_clusters, method):
    if method== 'kmeans':
        cluster_result = clusterKMeans(data, num_clusters)

        return -1*(metrics.silhouette_score(data, cluster_result.labels_, metric='euclidean'))

    if method == 'em':
        cluster_result = clusterEM(data, num_clusters)

        return -1*(metrics.silhouette_score(data, cluster_result.means_, metric = 'euclidean'))


def clusterKMeans(data, num_clusters):
    kmeans = KMeans(n_clusters = num_clusters).fit(data)
    return kmeans


def clusterEM(data, num_clusters):
    gmm = GaussianMixture(n_components = num_clusters).fit(data)
    return gmm
    

def clusterVI(data):
    print('TODO')

class Bayes_Optimal_Clusters():

    def __init__(self, domain, data, method, n_iter):
        self.domain = domain
        self.data = data
        self.method = method 
        self.noise = 0.1
        self.n_iter = n_iter
        self.num_clusters = None

        initial_x = np.linspace(2, self.domain, num=5)
        initial_x = initial_x.reshape(-1,1)
        initial_y = np.asarray([cluster_and_score(data, int(i), method) for i in initial_x ])
        initial_y = initial_y.reshape(-1,1)
        m52 = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5)
        self.gpmodel = GaussianProcessRegressor(kernel=m52, alpha = self.noise**2)

    
        self.x_sample, self.y_sample = initial_x, initial_y
        for i in range(n_iter):
            self.x_next = self.propose_location(self.gpmodel, self.domain)
            self.y_next = cluster_and_score(self.data, int(self.x_next), self.method)
            self.x_sample = np.vstack((self.x_sample, self.x_next))
            self.y_sample = np.vstack((self.y_sample, self.y_next))
            self.gpmodel.fit(self.x_sample, self.y_sample)
        
        self.num_clusters = self.x_next
        

    def lower_confidence_bound(self, x, gpmodel, kappa=2):
        mu, sigma = gpmodel.predict(np.full(1,x).reshape(-1,1), return_std = True)
        return mu-kappa*sigma

    def propose_location(self, gpmodel, domain, n_restarts = 25):
        dim = 1
        min_val = 1
        min_x = None
        def min_obj(X):
            return self.lower_confidence_bound(X, gpmodel)

        for x0 in np.random.uniform(2, domain, size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds = ((2,domain),),  method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x
        return round(min_x[0])


class Brute_Force_Optimal_Clusters():

    def __init__(self, domain, data, method):
        self.domain = domain
        self.data = data
        self.method = method
        self.num_clusters = None
        self.result = 1
        for i in range(2, self.domain):
            new_result = cluster_and_score(self.data, i, self.method)
            if new_result < self.result:
                self.result = new_result
                self.num_clusters = i
        
if __name__ == "__main__":
    true_clusters = 40
    sample_data_points = 100000
    data_gen = Data_Generator(true_clusters, sample_data_points)
    data = data_gen.data
    start_bayes_opt = time.time()
    opt_clusters_bayes_opt = Bayes_Optimal_Clusters(domain = 70, data = data, method = 'kmeans', n_iter = 20).num_clusters
    stop_bayes_opt = time.time()

    start_brute_force = time.time()
    opt_clusters_brute_force = Brute_Force_Optimal_Clusters(domain = 70, data = data, method = 'kmeans').num_clusters
    stop_brute_force = time.time()

    print('True number of clusters is {}'.format(true_clusters))
    print('Brute force found an optimal number of clusters {} in {:0.1f} seconds'.format(opt_clusters_brute_force, stop_brute_force-start_brute_force))
    print('Bayes optimization found an optimal number of clusters {} in {:0.1f} seconds'.format(int(opt_clusters_bayes_opt), stop_bayes_opt - start_bayes_opt))