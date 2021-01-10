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
from pyclustering.cluster.xmeans import xmeans, splitting_type
import scipy.cluster.hierarchy as h
from experiment.visualize import Visual
from experiment.clustering import biKmeans
from sklearn.metrics import silhouette_score ,calinski_harabasz_score,davies_bouldin_score
from scipy.spatial import distance
import json
from collect import patch_bert_vector
from experiment.evaluate import evaluation
import math

class Experiment:
    def __init__(self, path_test, path_patch_root, path_collected_patch, path_test_function_patch_vector, patch_w2v):
        self.path_test = path_test
        # self.path_test_vector = path_test_vector
        # self.path_patch_vector = path_patch_vector
        self.path_patch_root = path_patch_root
        self.path_collected_patch = path_collected_patch

        self.path_test_function_patch_vector = path_test_function_patch_vector
        self.patch_w2v = patch_w2v

        self.test_data = None
        # self.patch_data = None

        self.test_name = None
        self.patch_name = None
        self.test_vector = None
        self.patch_vector = None
        self.exception_type = None

    def load_test(self,):
        with open(self.path_test, 'rb') as f:
            self.test_data = pickle.load(f)
        if os.path.exists(path_test_function_patch_vector):
            both_vector = pickle.load(open(self.path_test_function_patch_vector, 'rb'))
            self.test_name = both_vector[0]
            self.patch_name = both_vector[1]
            self.test_vector = both_vector[2]
            self.patch_vector = both_vector[3]
            self.exception_type = both_vector[4]

        else:
            # test_vector = self.test2vector(word2v='code2vec')
            # patch_vector = self.patch2vector(word2v='cc2vec')
            all_test_name, all_patch_name, all_test_vector, all_patch_vector, all_exception_type = self.test_patch_2vector(test_w2v='code2vec', patch_w2v=self.patch_w2v)

            both_vector = [all_test_name, all_patch_name, all_test_vector, all_patch_vector, all_exception_type]
            pickle.dump(both_vector, open(self.path_test_function_patch_vector, 'wb'))
            # np.save(self.path_test_function_patch_vector, both_vector)

            self.test_name = both_vector[0]
            self.patch_name = both_vector[1]
            self.test_vector = both_vector[2]
            self.patch_vector = both_vector[3]
            self.exception_type = both_vector[4]


    def test_patch_2vector(self, test_w2v='code2vec', patch_w2v='cc2vec'):
        all_test_name, all_patch_name, all_test_vector, all_patch_vector, all_exception_type = [], [], [], [], []
        w2v = Word2vector(test_w2v=test_w2v, patch_w2v=patch_w2v, path_patch_root=self.path_patch_root)

        test_name_list = self.test_data[0]
        exception_type_list = self.test_data[1]
        log_list = self.test_data[2]
        test_function_list = self.test_data[3]
        patch_ids_list = self.test_data[4]
        for i in range(len(test_name_list)):
            name = test_name_list[i]
            function = test_function_list[i]
            ids = patch_ids_list[i]

            exception_type = exception_type_list[i]
            # if ':' in exception_type:
            #     exception_type = exception_type.split(':')[0]

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
            all_exception_type.append(exception_type)
            if len(all_test_vector) != len(all_patch_vector):
                print('???')

        if self.patch_w2v == 'string':
            return all_test_name, all_patch_name, np.array(all_test_vector), all_patch_vector, all_exception_type
        else:
            return all_test_name, all_patch_name, np.array(all_test_vector), np.array(all_patch_vector), all_exception_type

    def test2vector(self, word2v='code2vec'):
        w2v = Word2vector(test_w2v=word2v, path_patch_root=self.path_patch_root)
        test_function = self.test_data[3]
        test_name = self.test_data[0]
        self.test_vector = w2v.convert(test_name, test_function)
        print('test vector done')
        return self.test_vector
        # np.save(self.path_test_vector, self.test_vector)

    def patch2vector(self, word2v='cc2vec'):
        w2v = Word2vector(patch_w2v=word2v, path_patch_root=self.path_patch_root)
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

    def patch_dist(self, patch_vector, clusters, method_cluster, number):
        scaler = Normalizer()
        P = pd.DataFrame(scaler.fit_transform(patch_vector))
        P["Cluster"] = clusters

        if number <= 6:
            v = Visual(algorithm='PCA', number_cluster=number, method=method_cluster)
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



    def evaluate_defects4j_projects(self, ):
        scaler = Normalizer()
        all_test_vector = scaler.fit_transform(self.test_vector)
        scaler_patch = scaler.fit(self.patch_vector)
        all_patch_vector = scaler_patch.transform(self.patch_vector)
        projects = {'Chart': 26, 'Lang': 65, 'Time': 27, 'Closure': 176, 'Math': 106, 'Cli': 40, 'Codec': 18, 'Collections': 28, 'Compress': 47, 'Csv': 16, 'Gson': 18, 'JacksonCore': 26, 'JacksonDatabind': 112, 'JacksonXml': 6, 'Jsoup': 93, 'JxPath': 22, 'Mockito': 38}
        # projects = {'Mockito': 38}
        for project, number in projects.items():
            project_list = []
            print('Testing {}'.format(project))
            cnt = 0
            for i in range(len(self.test_name)):
                if not self.test_name[i].startswith(project):
                    continue
                project = self.test_name[i].split('-')[0].split('_')[0]
                id = self.test_name[i].split('-')[0].split('_')[1]
                print('{}'.format(self.test_name[i]))
                this_test = all_test_vector[i]
                this_patch = all_patch_vector[i]

                dist_min_index = None
                dist_min = 9999
                for j in range(len(all_test_vector)):
                    if j == i:
                        continue
                    # whether skip current project
                    if self.test_name[j].startswith(project+'_'+id+'-'):
                        continue
                    dist = distance.euclidean(this_test, all_test_vector[j])/(1 + distance.euclidean(this_test, all_test_vector[j]))
                    if dist < dist_min:
                        dist_min = dist
                        dist_min_index = j
                print('the closest test: {}'.format(self.test_name[dist_min_index]))
                closest_patch = all_patch_vector[dist_min_index]

                distance_patch = distance.euclidean(closest_patch, this_patch)/(1 + distance.euclidean(closest_patch, this_patch))
                # distance_patch = distance.cosine(closest_patch, this_patch)

                score_patch = 1 - distance_patch
                if math.isnan(score_patch):
                    continue
                project_list.append([self.test_name[i], score_patch])
            if project_list == []:
                print('{} no found'.format(project))
                continue
            recommend_list_project = pd.DataFrame(sorted(project_list, key=lambda x: x[1], reverse=True))
            # plt.bar(recommend_list_project.index.tolist(), recommend_list_project[:][1], color='chocolate')
            plt.bar(recommend_list_project.index.tolist(), recommend_list_project[:][1], color='steelblue')
            plt.xlabel('failed test cases')
            plt.ylabel('score of patch from the closest test case')
            plt.title('score distribution of {}'.format(project))
            plt.savefig('../fig/RQ2/distance_patch_{}'.format(project))
            plt.cla()

    # def evaluate_patch_sim(self, testdata):

    def run(self):

        # load data and vector
        self.load_test()

        # pre-save bert vector for test dataset
        # patch_bert_vector.patch_bert()


        # validate hypothesis
        # method_cluster = 'biKmeans'
        # number_cluster = 40
        # clusters = self.cluster_test_dist(self.test_vector, method=method_cluster, number=number_cluster)
        # self.patch_dist(self.patch_vector, clusters, method_cluster, number_cluster)

        # evaluate on developer's patch of defects4j
        # self.evaluate_defects4j_projects()

        # evaluate collected patches for projects
        eval = evaluation(self.patch_w2v, self.test_data, self.test_name, self.test_vector, self.patch_vector, self.exception_type)
        # eval.evaluate_collected_projects(self.path_collected_patch)
        eval.predict_collected_projects(self.path_collected_patch)

        # evaluate patch sim dataset
        # testdata = '/Users/haoye.tian/Documents/University/data/PatchSimISSTA_sliced/'
        # self.evaluate_collected_projects(testdata)

if __name__ == '__main__':
    config = Config()
    path_test = config.path_test
    path_patch_root = config.path_patch_root
    path_collected_patch = config.path_collected_patch

    path_test_function_patch_vector = config.path_test_function_patch_vector
    patch_w2v = config.patch_w2v

    e = Experiment(path_test, path_patch_root, path_collected_patch, path_test_function_patch_vector, patch_w2v)
    e.run()