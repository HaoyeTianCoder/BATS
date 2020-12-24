import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '.')))
# import seaborn as sns
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from matplotlib.pyplot import boxplot
from experiment.config import Config
from representation.word2vector import Word2vector
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from pyclustering.cluster.xmeans import xmeans
import scipy.cluster.hierarchy as h
from experiment.visualize import Visual

class Experiment:
    def __init__(self, path_test, path_patch_root, path_test_vector):
        self.path_test = path_test
        self.path_test_vector = path_test_vector
        self.path_patch_root = path_patch_root

        self.test_data = None
        self.patch_data = None
        self.test_vector = None
        self.patch_vector = None

    def load_test(self,):
        if os.path.exists(self.path_test_vector):
            self.test_vector = np.load(self.path_test_vector)
            print('test vector detected!')
            return
        else:
            with open(self.path_test,'rb') as f:
                self.test_data = pickle.load(f)
            # self.patch_data = pickle.load(self.path_patch)
            self.test2vector(word2v='code2vec')
            self.patch2vector(word2v='cc2vec')

    def run(self):
        self.load_test()

        # dists_one = self.cal_all_simi(self.test_vector)
        result_cluster = self.cluster_dist(self.test_vector, method='dbscan')
        plt.boxplot(result_cluster, labels=['Original']+['Cluster'+str(i) for i in range(len(result_cluster)-1)] )
        # plt.boxplot([dists_one, dist0, dist1, dist2, ], labels=['Original', 'Cluster1', 'Cluster2', 'Cluster3'])
        plt.xlabel('Cluster')
        plt.ylabel('Distance to Center')
        plt.show()


    def test2vector(self, word2v='code2vec'):
        w2v = Word2vector(word2v, self.path_patch_root)
        test_function = self.test_data[3]
        test_name = self.test_data[0]
        self.test_vector = w2v.convert(test_name, test_function)
        print('test vector done')
        np.save(self.path_test_vector, self.test_vector)

    def patch2vector(self, word2v='cc2vec'):
        w2v = Word2vector(word2v, self.path_patch_root)
        # find corresponding patch id through test case pickle
        test_name = self.test_data[0]
        patch_id = self.test_data[4]
        self.patch_vector = w2v.convert(test_name, patch_id)

    def cal_all_simi(self, test_vector):
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(test_vector))

        center = np.mean(X, axis=0)
        dists_one = [np.linalg.norm(vec - center) for vec in np.array(X)]
        # average = np.array(dists).mean()
        # print('one cluster average distance: {}'.format(average))
        # plt.boxplot(dists,)
        # ax = sns.boxplot(x="all", y="distance", data=dists)

        return dists_one

    # def cluster_dist(self, test_vector):
    #     dist0, dist1, dist2 = self.cluster(test_vector)
    #     return dist0, dist1, dist2

    def cluster_dist(self, test_vector, method):
        scaler = Normalizer()
        X = pd.DataFrame(scaler.fit_transform(test_vector))

        # one cluster
        center = np.mean(X, axis=0)
        dists_one = [np.linalg.norm(vec - center) for vec in np.array(X)]

        if method == 'kmeans':
            number_cluster = 4
            kmeans = KMeans(n_clusters=number_cluster)
            # kmeans.fit(np.array(test_vector))
            clusters = kmeans.fit_predict(X)
        elif method == 'dbscan':
            db = DBSCAN(eps=1, min_samples=10)
            clusters = db.fit_predict(X)
            number_cluster = max(clusters)+2
        elif method == 'hier':
            number_cluster = 3
            hu = AgglomerativeClustering(n_clusters=number_cluster)
            clusters = hu.fit_predict(X)
        elif method == 'xmeans':
            xmeans_instance = xmeans(X,)
            clusters = xmeans_instance.process().predict(X)
            number_cluster = max(clusters)+1
        elif method == 'ap':
            # ap = AffinityPropagation(random_state=5)
            # clusters = ap.fit_predict(X)

            APC = AffinityPropagation(verbose=True, max_iter=400, convergence_iter=25).fit(X)
            APC_res = APC.predict(X)
            clusters = APC.cluster_centers_indices_
        X["Cluster"] = clusters

        if number_cluster <= 6:
            algorithm = 'dbscan'
            v = Visual('PCA', number_cluster, algorithm=algorithm)
            v.visualize(plotX=X)

        result = [dists_one]
        for i in range(number_cluster):
            if method == 'dbscan':
                i -= 1
            cluster = X[X["Cluster"] == i].drop(["Cluster"], axis=1)
            center = np.mean(cluster, axis=0)
            dist = [np.linalg.norm(vec - center) for vec in np.array(cluster)]
            result.append(dist)

        return result




if __name__ == '__main__':
    config = Config()
    path_test = config.path_test
    path_patch_root = config.path_patch_root
    path_test_vector = config.path_test_vector

    e = Experiment(path_test, path_patch_root, path_test_vector)
    e.run()