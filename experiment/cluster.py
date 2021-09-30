import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '.')))
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from pyclustering.cluster.xmeans import xmeans, splitting_type
import scipy.cluster.hierarchy as h
from experiment.visualize import Visual
from experiment.clustering import biKmeans
from sklearn.metrics import silhouette_score ,calinski_harabasz_score, davies_bouldin_score
from scipy.spatial import distance
import scipy.stats as stats

class cluster:
    def __init__(self, original_dataset, test_name, patch_name,test_vector, patch_vector, method, number):
        self.original_dataset = original_dataset
        self.test_name = test_name
        self.patch_name = patch_name

        self.test_vector = test_vector
        self.patch_vector = patch_vector
        self.method = method
        self.number = number

    def validate(self,):
        print('Research Question 1')
        scaler = Normalizer()
        clusters = self.cluster_test_dist(method=self.method, number=self.number, scaler=scaler)
        self.patch_dist(clusters, self.method, self.number, scaler=scaler)

        # print('Random or Independent **************')
        # self.patch_dist(self.patch_vector, [i for i in range(40)] + [random.randint(0, 40) for j in range(418)], self.method, self.number, scaler=scaler)
        # self.patch_dist([i for i in range(40)] + [random.randint(0, 40) for j in range(1080)], self.method, self.number, scaler=scaler)
        # self.cluster_patch_dist_inde(self.method, self.number, scaler)

        # reverse test case and patch
        # clusters = self.cluster_patch_dist_inde(self.method, self.number, scaler)
        # self.test_dist(clusters, self.method, self.number, scaler=scaler)

    def cluster_test_dist(self, method, number, scaler):
        # scaler = Normalizer()
        X = pd.DataFrame(scaler.fit_transform(self.test_vector))

        # original distance as one cluster
        center_one = np.mean(X, axis=0)
        dists_one = [distance.euclidean(vec, np.array(center_one)) for vec in np.array(X)]

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
        else:
            raise
        # X["Cluster"] = clusters

        # s1 = silhouette_score(X, clusters)
        # s2 = calinski_harabasz_score(X, clusters)
        # s3 = davies_bouldin_score(X, clusters)
        print('TEST------')
        # print('Silhouette: {}'.format(s1))
        # print('CH: {}'.format(s2))
        # print('DBI: {}'.format(s3))

        self.score_inside_outside(X, clusters, number)
        return clusters

        # visualize clustering.
        # X["Cluster"] = clusters
        # if number <= 6:
        #     v = Visual(algorithm='PCA', number_cluster=number, method=method)
        #     v.visualize(plotX=X)

    '''
        # boxplot of distance to center in each cluster
        result_cluster = [dists_one]
        for i in range(number):
            if method == 'dbscan':
                i -= 1
            cluster = X[X["Cluster"] == i].drop(["Cluster"], axis=1)
            center = np.mean(cluster, axis=0)
            # dist = [np.linalg.norm(vec - np.array(center)) for vec in np.array(cluster)]
            dist = [distance.euclidean(vec, np.array(center)) for vec in np.array(cluster)]
            result_cluster.append(dist)

        # boxplot figure covering two columns
        plt.figure(figsize=(18, 8))
        plt.xticks(fontsize=15, )
        plt.yticks(fontsize=15, )
        bplot = plt.boxplot(result_cluster, labels=['All']+[str(i) for i in range(len(result_cluster)-1)], widths=0.4, patch_artist=True)
        colors = ['red']+['white' for i in range(len(bplot['boxes'])-1)]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.xlabel('Cluster', fontsize=17)
        plt.ylabel('Euclidian distance', fontsize=17)
        plt.savefig('../fig/RQ1/boxplot.png')
    '''


    def cluster_patch_dist_inde(self, method, number, scaler):
        # scaler = Normalizer()
        # scaler = MinMaxScaler()
        P = pd.DataFrame(scaler.fit_transform(self.patch_vector))

        # one cluster
        center_one = np.mean(P, axis=0)
        dists_one = [np.linalg.norm(vec - np.array(center_one)) for vec in np.array(P)]

        if method == 'kmeans':
            kmeans = KMeans(n_clusters=number, random_state=1)
            # kmeans.fit(np.array(test_vector))
            clusters = kmeans.fit_predict(P)
        elif method == 'dbscan':
            db = DBSCAN(eps=0.5, min_samples=10)
            clusters = db.fit_predict(P)
            number = max(clusters)+2
        elif method == 'hier':
            hu = AgglomerativeClustering(n_clusters=number)
            clusters = hu.fit_predict(P)
        elif method == 'xmeans':
            xmeans_instance = xmeans(P, kmax=200, splitting_type=splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH)
            clusters = xmeans_instance.process().predict(P)
            # clusters = xmeans_instance.process().get_clusters()
            number = max(clusters)+1
        elif method == 'biKmeans':
            bk = biKmeans()
            clusters = bk.biKmeans(dataSet=np.array(P), k=number)
        elif method == 'ap':
            # ap = AffinityPropagation(random_state=5)
            # clusters = ap.fit_predict(X)
            APC = AffinityPropagation(verbose=True, max_iter=200, convergence_iter=25).fit(P)
            APC_res = APC.predict(P)
            clusters = APC.cluster_centers_indices_
        else:
            raise
        # X["Cluster"] = clusters

        # s1 = silhouette_score(P, clusters)
        # s2 = calinski_harabasz_score(P, clusters)
        # s3 = davies_bouldin_score(P, clusters)
        print('Patch independently------')
        # print('Silhouette: {}'.format(s1))
        # print('CH: {}'.format(s2))
        # print('DBI: {}'.format(s3))

        self.score_inside_outside(P, clusters, number)
        return clusters

    def test_dist(self, clusters, method, number, scaler):
        X = pd.DataFrame(scaler.fit_transform(self.test_vector))
        # P = pd.DataFrame(scaler.fit_transform(self.patch_vector))
        # P["Cluster"] = clusters

        # if number <= 6:
        #     v = Visual(algorithm='PCA', number_cluster=number, method=method)
        #     v.visualize(plotX=P)

        # s1 = silhouette_score(P, clusters)
        # s2 = calinski_harabasz_score(P, clusters)
        # s3 = davies_bouldin_score(P, clusters)
        print('TEST------')
        # print('Silhouette: {}'.format(s1))
        # print('CH: {}'.format(s2))
        # print('DBI: {}'.format(s3))

        # loose metric
        self.score_inside_outside(X, clusters, number)

    def patch_dist(self, clusters, method, number, scaler):
        # X = pd.DataFrame(scaler.fit_transform(self.test_vector))
        P = pd.DataFrame(scaler.fit_transform(self.patch_vector))
        # P["Cluster"] = clusters

        if number <= 6:
            v = Visual(algorithm='PCA', number_cluster=number, method=method)
            v.visualize(plotX=P)

        # s1 = silhouette_score(P, clusters)
        # s2 = calinski_harabasz_score(P, clusters)
        # s3 = davies_bouldin_score(P, clusters)
        print('PATCH------')
        # print('Silhouette: {}'.format(s1))
        # print('CH: {}'.format(s2))
        # print('DBI: {}'.format(s3))

        # loose metric
        self.score_inside_outside(P, clusters, number)

        '''
        MWW validation
        cnt = 0
        for n in range(number):
            index = np.where(clusters==n)

            test_mww = []
            test_vector = X.iloc[index]
            test_center = np.mean(test_vector, axis=0)
            for i in test_vector.values:
                dist = distance.euclidean(i, test_center)/ (1+distance.euclidean(i, test_center))
                score = 1 - dist
                test_mww.append(score)

            patch_mww = []
            patch_vector = P.iloc[index]
            patch_center = np.mean(patch_vector, axis=0)
            for j in patch_vector.values:
                dist = distance.euclidean(j, patch_center)/ (1+distance.euclidean(j, patch_center))
                score = 1 - dist
                patch_mww.append(score)
            try:
                hypo = stats.mannwhitneyu(test_mww, patch_mww, alternative='two-sided')
                p_value = hypo[1]
            except Exception as e:
                if 'identical' in e:
                    p_value = 1
            print('p-value: {}'.format(p_value))
            if p_value >= 0.05:
                cnt += 1
        print('{}/{} satisfied the hypothesis'.format(cnt, number))
        '''

        # n = 1
        # index = np.where(clusters==n)
        # patch_name = np.array(self.patch_name)[index]
        # test_name = np.array(self.test_name)[index]
        # function_name = np.array(self.original_dataset[3])[index]
        # print('cluster {}'.format(n))
        # for i in range(len(test_name)):
        #     print('test&patch:{}'.format(test_name[i]), end='    ')
        #     print('{}'.format(patch_name[i]))
        #
        #     # print('function:{}'.format(function_name[i]))

    def score_inside_outside(self, vectors, clusters, number):
        cnt = 0
        diffs = []
        print('Calculating...')
        for n in range(number):
            # print('cluster index: {}'.format(n))
            index_inside = np.where(clusters == n)
            score_inside_mean = []
            score_outside_mean = []
            vector_inside = vectors.iloc[index_inside]
            for i in range(vector_inside.shape[0]):
                cur = vector_inside.iloc[i]

                # compared to vectors inside this cluster
                for j in range(i+1, vector_inside.shape[0]):
                    cur2 = vector_inside.iloc[j]
                    dist = distance.euclidean(cur, cur2) / (1 + distance.euclidean(cur, cur2))
                    score = 1 - dist
                    score_inside_mean.append(score)

                # compared to vectors outside the cluster
                index_outside = np.where(clusters!=n)
                vector_outside = vectors.iloc[index_outside]
                for k in range(vector_outside.shape[0]):
                    cur3 = vector_outside.iloc[k]
                    dist = distance.euclidean(cur, cur3) / (1 + distance.euclidean(cur, cur3))
                    score = 1 - dist
                    score_outside_mean.append(score)

            inside_score = np.array(score_inside_mean).mean()
            outside_score = np.array(score_outside_mean).mean()
            # print('inside: {}'.format(inside_score), end='    ')
            # print('outside: {}'.format(outside_score))

            SC = (inside_score - outside_score) / max(inside_score, outside_score)
            diffs.append(SC)

        CSC = np.array(diffs).mean()
        print('Qualified: {}/{}'.format(len(np.where(np.array(diffs)>0)[0]), len(diffs)))
        print('CSC: {}'.format(CSC))
