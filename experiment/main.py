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
from sklearn.metrics import silhouette_score ,calinski_harabasz_score,davies_bouldin_score

class Experiment:
    def __init__(self, path_test, path_patch_root, path_test_function_patch_vector):
        self.path_test = path_test
        # self.path_test_vector = path_test_vector
        # self.path_patch_vector = path_patch_vector
        self.path_patch_root = path_patch_root

        self.path_test_function_patch_vector = path_test_function_patch_vector

        self.test_data = None
        # self.patch_data = None

        self.test_name = None
        self.patch_name = None
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

        with open(self.path_test, 'rb') as f:
            self.test_data = pickle.load(f)
        if os.path.exists(path_test_function_patch_vector):
            both_vector = pickle.load(open(self.path_test_function_patch_vector, 'rb'))
            self.test_name = both_vector[0]
            self.patch_name = both_vector[1]
            self.test_vector = both_vector[2]
            self.patch_vector = both_vector[3]
        else:
            # test_vector = self.test2vector(word2v='code2vec')
            # patch_vector = self.patch2vector(word2v='cc2vec')
            all_test_name, all_patch_name, all_test_vector, all_patch_vector = self.test_patch_2vector(test_w2v='code2vec', patch_w2v='cc2vec')

            both_vector = [all_test_name, all_patch_name, all_test_vector, all_patch_vector]
            pickle.dump(both_vector, open(self.path_test_function_patch_vector, 'wb'))
            # np.save(self.path_test_function_patch_vector, both_vector)

            self.test_name = both_vector[0]
            self.patch_name = both_vector[1]
            self.test_vector = both_vector[2]
            self.patch_vector = both_vector[3]


    def test_patch_2vector(self, test_w2v='code2vec', patch_w2v='cc2vec'):
        all_test_name, all_patch_name, all_test_vector, all_patch_vector = [], [], [], []
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
            all_test_name.append(name)
            all_patch_name.append(ids)
            all_test_vector.append(test_vector)
            all_patch_vector.append(patch_vector)
            if len(all_test_vector) != len(all_patch_vector):
                print('???')

        return all_test_name, all_patch_name, np.array(all_test_vector), np.array(all_patch_vector)


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
            xmeans_instance = xmeans(X, random_state=1)
            clusters = xmeans_instance.process().predict(X)
            number = max(clusters)+1
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
        plt.savefig('../fig/box_{}.png'.format(method))

        return clusters

    def patch_dist(self, patch_vector, clusters, method_cluster, number):
        scaler = Normalizer()
        P = pd.DataFrame(scaler.fit_transform(patch_vector))
        P["Cluster"] = clusters

        if number <= 6:
            v = Visual(algorithm='PCA', number_cluster=number, method=method_cluster)
            v.visualize(plotX=P)

        s1 = silhouette_score(P, clusters, metric='euclidean')
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

            print('function:{}'.format(function_name[i]))

    def predict(self, failed_test_index, new_patch):
        scaler = Normalizer()
        all_test_vector = scaler.fit_transform(self.test_vector)
        all_patch_vector = scaler.fit_transform(self.patch_vector)

        k = 10
        dataset_test = np.delete(all_test_vector, failed_test_index, axis=0)
        dataset_patch = np.delete(all_patch_vector, failed_test_index, axis=0)
        for i in failed_test_index:
            failed_test_vector = all_test_vector[i]
            dists = []
            # find the k most closest test vector
            for i in range(len(dataset_test)):
                simi_test_vec = dataset_test[i]
                dist = np.linalg.norm(simi_test_vec - failed_test_vector)
                dists.append([i, dist])
            k_index = sorted(dists, key=lambda x:int(x[1]), reverse=False)[:k]
            k_index = np.array([v[0] for v in k_index])

            k_patch_vector = dataset_patch[k_index]
            dist_k = []
            for i in range(len(k_patch_vector)):
                vec = k_patch_vector[i]
                dist = np.linalg.norm(vec - new_patch)
                dist_k.append(dist)

            dist_mean = np.array(dist_k).mean()
            dist_min = np.array(dist_k).min()

            print('mean:{}  min:{}'.format(dist_mean, dist_min))





    def run(self):
        # load data and vector
        self.load_test()

        '''
        # validate hypothesis
        method_cluster = 'kmeans'
        number_cluster = 40
        clusters = self.cluster_test_dist(self.test_vector, method=method_cluster, number=number_cluster)
        self.patch_dist(self.patch_vector, clusters, method_cluster, number_cluster)
        '''

        # predict
        project_id = 'Closure_49'
        failed_test_index = [i for i in range(len(self.test_name)) if self.test_name[i].startswith(project_id)]
        new_patch = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1/ConFix/Incorrect/Math/56'
        label = 'Correct'
        self.predict(failed_test_index, new_patch)

if __name__ == '__main__':
    config = Config()
    path_test = config.path_test
    # path_test_vector = config.path_test_vector

    path_patch_root = config.path_patch_root
    # path_patch_vector = config.path_patch_vector

    path_test_function_patch_vector = config.path_test_function_patch_vector

    e = Experiment(path_test, path_patch_root, path_test_function_patch_vector)
    e.run()