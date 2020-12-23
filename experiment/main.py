import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '.')))
# import seaborn as sns
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.pyplot import boxplot
from experiment.config import Config
from representation.word2vector import Word2vector
import numpy as np
from sklearn.cluster import KMeans
from visualize import Visual

class Experiment:
    def __init__(self, path_test, path_patch):
        self.path_test = path_test
        self.path_patch = path_patch

        self.test_data = None
        self.patch_data = None
        self.test_vector = None
        self.patch_vector = None

    def load_test(self,):
        with open(self.path_test,'rb') as f:
            self.test_data = pickle.load(f)

        # self.patch_data = pickle.load(self.path_patch)

    def run(self):
        self.load_test()

        self.test2vector(word2v='code2vec')
        # self.patch2vector(word2v='cc2vec')

        # dists_one = self.cal_all_simi(self.test_vector)
        dists_one, dist0, dist1, dist2, dist3 = self.cluster_dist(self.test_vector)
        plt.boxplot([dists_one, dist0, dist1, dist2, dist3],labels=['Original','Cluster1','Cluster2','Cluster3','Cluster4'] )
        plt.boxplot([dists_one, dist0, dist1, dist2, ], labels=['Original', 'Cluster1', 'Cluster2', 'Cluster3'])
        plt.xlabel('Cluster')
        plt.ylabel('Distance to Center')



    def test2vector(self, word2v='code2vec'):
        w2v = Word2vector(word2v)
        test_function = self.test_data[3]
        test_name = self.test_data[0]
        self.test_vector = w2v.convert(test_name, test_function)
        print('test vector done')

    def patch2vector(self, word2v='cc2vec'):
        w2v = Word2vector(word2v)
        self.patch_vector = w2v.convert(self.patch_data)

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

    def cluster_dist(self, test_vector):
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(test_vector))

        # one cluster
        center = np.mean(X, axis=0)
        dists_one = [np.linalg.norm(vec - center) for vec in np.array(X)]

        kmeans = KMeans(n_clusters=4)
        # kmeans.fit(np.array(test_vector))
        clusters = kmeans.fit_predict(X)
        X["Cluster"] = clusters

        # v = Visual('PCA')
        # v.visualize(clusters, plotX=X)

        cluster0 = X[X["Cluster"] == 0].drop(["Cluster"], axis=1)
        center0 = np.mean(cluster0, axis=0)
        dist0 = [np.linalg.norm(vec - center0) for vec in np.array(cluster0)]

        cluster1 = X[X["Cluster"] == 1].drop(["Cluster"], axis=1)
        center1 = np.mean(cluster1, axis=0)
        dist1 = [np.linalg.norm(vec - center1) for vec in np.array(cluster1)]

        cluster2 = X[X["Cluster"] == 2].drop(["Cluster"], axis=1)
        center2 = np.mean(cluster2, axis=0)
        dist2 = [np.linalg.norm(vec - center2) for vec in np.array(cluster2)]

        cluster3 = X[X["Cluster"] == 3].drop(["Cluster"], axis=1)
        center3 = np.mean(cluster3, axis=0)
        dist3 = [np.linalg.norm(vec - center3) for vec in np.array(cluster3)]

        return dists_one, dist0, dist1, dist2, dist3




if __name__ == '__main__':
    config = Config()
    path_test = config.path_test
    path_patch = config.path_patch

    e = Experiment(path_test, path_patch)
    e.run()