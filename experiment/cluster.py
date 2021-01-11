import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '.')))
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from pyclustering.cluster.xmeans import xmeans, splitting_type
import scipy.cluster.hierarchy as h
from experiment.visualize import Visual
from experiment.clustering import biKmeans
from sklearn.metrics import silhouette_score ,calinski_harabasz_score,davies_bouldin_score


class cluster:
    def __init__(self, test_data, test_name, patch_name,test_vector, patch_vector, method, number):
        self.test_data = test_data
        self.test_name = test_name
        self.patch_name = patch_name

        self.test_vector = test_vector
        self.patch_vector = patch_vector
        self.method = method
        self.number = number

    def validate(self,):
        clusters = self.cluster_test_dist(self.test_vector, method=self.method, number=self.number)
        self.patch_dist(self.patch_vector, clusters, self.method, self.number)

    def cluster_test_dist(self, test_vector, method, number):
        scaler = Normalizer()
        X = pd.DataFrame(scaler.fit_transform(test_vector))

        # one cluster
        center_one = np.mean(X, axis=0)
        dists_one = [np.linalg.norm(vec - np.array(center_one)) for vec in np.array(X)]

        if method == 'kmeans':
            kmeans = KMeans(n_clusters=number, random_state=1)
            # kmeans.fit(np.array(test_vector))
            clusters = kmeans.fit_predict(X)
        elif method == 'dbscan':
            db = DBSCAN(eps=0.5, min_samples=10)
            clusters = db.fit_predict(X)
            number = max(clusters)+2
        elif method == 'hier':
            hu = AgglomerativeClustering(n_clusters=number)
            clusters = hu.fit_predict(X)
        elif method == 'xmeans':
            xmeans_instance = xmeans(X, kmax=200, splitting_type=splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH)
            clusters = xmeans_instance.process().predict(X)
            # clusters = xmeans_instance.process().get_clusters()
            number = max(clusters)+1
        elif method == 'biKmeans':
            bk = biKmeans()
            clusters = bk.biKmeans(dataSet=np.array(X), k=number)
        elif method == 'ap':
            # ap = AffinityPropagation(random_state=5)
            # clusters = ap.fit_predict(X)
            APC = AffinityPropagation(verbose=True, max_iter=200, convergence_iter=25).fit(X)
            APC_res = APC.predict(X)
            clusters = APC.cluster_centers_indices_
        X["Cluster"] = clusters

        s1 = silhouette_score(X, clusters, metric='euclidean')
        s2 = calinski_harabasz_score(X, clusters)
        s3 = davies_bouldin_score(X, clusters)
        print('TEST------')
        print('Silhouette: {}'.format(s1))
        print('CH: {}'.format(s2))
        print('DBI: {}'.format(s3))

        if number <= 6:
            v = Visual(algorithm='PCA', number_cluster=number, method=method)
            v.visualize(plotX=X)

        result_cluster = [dists_one]
        for i in range(number):
            if method == 'dbscan':
                i -= 1
            cluster = X[X["Cluster"] == i].drop(["Cluster"], axis=1)
            center = np.mean(cluster, axis=0)
            dist = [np.linalg.norm(vec - np.array(center)) for vec in np.array(cluster)]
            result_cluster.append(dist)

        plt.boxplot(result_cluster, labels=['Original']+[str(i) for i in range(len(result_cluster)-1)] )
        plt.xlabel('Cluster')
        plt.ylabel('Distance to Center')
        plt.savefig('../fig/RQ1/box_{}.png'.format(method))

        return clusters

    def patch_dist(self, patch_vector, clusters, method, number):
        scaler = Normalizer()
        P = pd.DataFrame(scaler.fit_transform(patch_vector))
        P["Cluster"] = clusters

        if number <= 6:
            v = Visual(algorithm='PCA', number_cluster=number, method=method)
            v.visualize(plotX=P)

        s1 = silhouette_score(P, clusters, metric='cosine')
        s2 = calinski_harabasz_score(P, clusters)
        s3 = davies_bouldin_score(P, clusters)
        print('PATCH------')
        print('Silhouette: {}'.format(s1))
        print('CH: {}'.format(s2))
        print('DBI: {}'.format(s3))

        n = 1
        index = np.where(clusters==n)
        patch_name = np.array(self.patch_name)[index]
        test_name = np.array(self.test_name)[index]
        function_name = np.array(self.test_data[3])[index]

        print('cluster {}'.format(n))
        for i in range(len(test_name)):
            print('test&patch:{}'.format(test_name[i]), end='    ')
            print('{}'.format(patch_name[i]))

            # print('function:{}'.format(function_name[i]))