import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '.')))
# import seaborn as sns
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
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
    def __init__(self, path_test, path_patch_root, path_test_function_patch_vector):
        self.path_test = path_test
        # self.path_test_vector = path_test_vector
        # self.path_patch_vector = path_patch_vector
        self.path_patch_root = path_patch_root

        self.path_test_function_patch_vector = path_test_function_patch_vector

        self.test_data = None
        # self.patch_data = None

        self.test_vector = None
        self.patch_vector = None

    def load_test(self,):
        # if os.path.exists(self.path_test_vector) and os.path.exists(self.path_patch_vector):
        #     self.test_vector = np.load(self.path_test_vector)
        #     self.patch_vector = np.load(self.path_patch_vector)
        #     print('test and patch vector detected!')
        #     return
        # else:
        #     with open(self.path_test, 'rb') as f:
        #         self.test_data = pickle.load(f)
        #
        #     if os.path.exists(self.path_test_vector):
        #         self.test_vector = np.load(self.path_test_vector)
        #     else:
        #         self.test2vector(word2v='code2vec')
        #
        #     self.patch2vector(word2v='cc2vec')

        if os.path.exists(path_test_function_patch_vector):
            both_vector = pickle.load(open(self.path_test_function_patch_vector, 'rb'))
            self.test_vector = both_vector[0]
            self.patch_vector = both_vector[1]
        else:
            with open(self.path_test, 'rb') as f:
                self.test_data = pickle.load(f)

            # test_vector = self.test2vector(word2v='code2vec')
            # patch_vector = self.patch2vector(word2v='cc2vec')
            all_test_vector, all_patch_vector = self.test_patch_2vector(test_w2v='code2vec', patch_w2v='cc2vec')

            both_vector = list([all_test_vector, all_patch_vector])
            pickle.dump(both_vector, open(self.path_test_function_patch_vector, 'wb'))
            # np.save(self.path_test_function_patch_vector, both_vector)

    def run(self):
        self.load_test()

        # dists_one = self.cal_all_simi(self.test_vector)
        method_cluster = 'kmeans'
        result_cluster = self.cluster_dist(self.test_vector, method=method_cluster)


    def test_patch_2vector(self, test_w2v='code2vec', patch_w2v='cc2vec'):
        all_test_vector, all_patch_vector = [], []
        w2v = Word2vector(test_w2v=test_w2v, patch_w2v=patch_w2v, path_patch_root=self.path_patch_root)

        test_name_list = self.test_data[0]
        test_function_list = self.test_data[3]
        patch_ids_list = self.test_data[4]
        for i in range(len(test_name_list)):
            name = test_name_list[i]
            function = test_function_list[i]
            ids = patch_ids_list[i]
            try:
                test_vector, patch_vector = w2v.convert_both(name, function, ids)
            except Exception as e:
                print('{} test name:{} exception:{}'.format(i, name, e))
                continue
            print('{} test name:{} success!'.format(i, name,))
            all_test_vector.append(test_vector)
            all_patch_vector.append(patch_vector)
            if len(all_test_vector) != len(all_patch_vector):
                print('???')

        return np.array(all_test_vector), np.array(all_patch_vector)


    def test2vector(self, word2v='code2vec'):
        w2v = Word2vector(word2v, self.path_patch_root)
        test_function = self.test_data[3]
        test_name = self.test_data[0]
        self.test_vector = w2v.convert(test_name, test_function)
        print('test vector done')
        return self.test_vector
        # np.save(self.path_test_vector, self.test_vector)

    def patch2vector(self, word2v='cc2vec'):
        w2v = Word2vector(word2v, self.path_patch_root)
        # find corresponding patch id through test case pickle
        test_name = self.test_data[0]
        patch_id = self.test_data[4]
        self.patch_vector = w2v.convert(test_name, patch_id)
        print('patch vector done')
        return self.patch_vector
        # np.save(self.path_patch_vector, self.patch_vector)

    def cal_all_simi(self, test_vector):
        scaler = Normalizer()
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
        center_one = np.mean(X, axis=0)
        dists_one = [np.linalg.norm(vec - np.array(center_one)) for vec in np.array(X)]

        if method == 'kmeans':
            number_cluster =4
            kmeans = KMeans(n_clusters=number_cluster, random_state=8)
            # kmeans.fit(np.array(test_vector))
            clusters = kmeans.fit_predict(X)
        elif method == 'dbscan':
            db = DBSCAN(eps=0.1, min_samples=10)
            clusters = db.fit_predict(X)
            number_cluster = max(clusters)+2
        elif method == 'hier':
            number_cluster = 4
            hu = AgglomerativeClustering(n_clusters=number_cluster)
            clusters = hu.fit_predict(X)
        elif method == 'xmeans':
            xmeans_instance = xmeans(X,)
            clusters = xmeans_instance.process().predict(X)
            number_cluster = max(clusters)+1
        elif method == 'ap':
            # ap = AffinityPropagation(random_state=5)
            # clusters = ap.fit_predict(X)
            APC = AffinityPropagation(verbose=True, max_iter=200, convergence_iter=25).fit(X)
            APC_res = APC.predict(X)
            clusters = APC.cluster_centers_indices_
        X["Cluster"] = clusters

        if number_cluster <= 6:
            v = Visual(algorithm='PCA', number_cluster=number_cluster, method=method)
            # v.visualize(plotX=X)

        result_cluster = [dists_one]
        for i in range(number_cluster):
            if method == 'dbscan':
                i -= 1
            cluster = X[X["Cluster"] == i].drop(["Cluster"], axis=1)
            center = np.mean(cluster, axis=0)
            dist = [np.linalg.norm(vec - np.array(center)) for vec in np.array(cluster)]
            result_cluster.append(dist)

        plt.boxplot(result_cluster, labels=['Original']+['Cluster'+str(i) for i in range(len(result_cluster)-1)] )
        plt.xlabel('Cluster')
        plt.ylabel('Distance to Center')
        plt.show()
        plt.savefig('../fig/box_{}.png'.format(method))

        return cluster

if __name__ == '__main__':
    config = Config()
    path_test = config.path_test
    # path_test_vector = config.path_test_vector

    path_patch_root = config.path_patch_root
    # path_patch_vector = config.path_patch_vector

    path_test_function_patch_vector = config.path_test_function_patch_vector

    e = Experiment(path_test, path_patch_root, path_test_function_patch_vector)
    e.run()