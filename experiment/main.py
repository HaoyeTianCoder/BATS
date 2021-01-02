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
from scipy.spatial import distance
import json
from collect import patch_bert_vector
import Levenshtein
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

            # print('tmp transfer patch')
            # new_patch_vector = list()
            # for i in range(len(self.patch_vector)):
            #     if self.patch_vector[i] != []:
            #         arr = np.array(self.patch_vector[i])
            #     else:
            #         arr = np.array([0 for i in range(2050)])
            #     new_patch_vector.append(arr)
            # self.patch_vector = np.array(new_patch_vector)

        else:
            # test_vector = self.test2vector(word2v='code2vec')
            # patch_vector = self.patch2vector(word2v='cc2vec')
            all_test_name, all_patch_name, all_test_vector, all_patch_vector = self.test_patch_2vector(test_w2v='code2vec', patch_w2v=self.patch_w2v)

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

        if self.patch_w2v == 'str':
            return all_test_name, all_patch_name, np.array(all_test_vector), all_patch_vector
        else:
            return all_test_name, all_patch_name, np.array(all_test_vector), np.array(all_patch_vector)

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

            # print('function:{}'.format(function_name[i]))

    def find_path_patch(self, path_patch_sliced, project_id):
        available_path_patch = []

        project = project_id.split('_')[0]
        id = project_id.split('_')[1]

        tools = os.listdir(path_patch_sliced)
        for label in ['Correct', 'Incorrect']:
            for tool in tools:
                path_bugid = os.path.join(path_patch_sliced, tool, label, project, id)
                if os.path.exists(path_bugid):
                    patches = os.listdir(path_bugid)
                    for p in patches:
                        path_patch = os.path.join(path_bugid, p)
                        if os.path.isdir(path_patch):
                            available_path_patch.append(path_patch)
        return available_path_patch

    def vector4patch(self, available_path_patch):
        vector_list = []
        label_list = []
        name_list = []
        for p in available_path_patch:
            # label
            if 'Correct' in p:
                label_list.append(1)
                label = 'Correct'
            elif 'Incorrect' in p:
                label_list.append(0)
                label = 'Incorrect'
            else:
                raise Exception('wrong label')

            # name
            tool = p.split('/')[-5]
            patchid = p.split('/')[-1]
            # name = tool + '-' + label + '-' + patchid
            name = tool[:3] + patchid.replace('patch','')
            name_list.append(name)

            # vector
            json_key = p + '_.json'
            if os.path.exists(json_key) and self.patch_w2v == 'bert':
                with open(json_key, 'r+') as f:
                    vector_str = json.load(f)
                    vector = np.array(list(map(float, vector_str)))
            else:
                w2v = Word2vector(patch_w2v=self.patch_w2v, )
                vector = w2v.convert_single_patch(p)
                vector_json = list(vector)
                vector_json = list(map(str, vector_json))
                with open(json_key, 'w+') as f:
                    jsonstr = json.dumps(vector_json, )
                    f.write(jsonstr)
            vector_list.append(vector)

        return name_list, label_list, vector_list

    def get_patch_list(self, failed_test_index, k=10, filt=False):
        scaler = Normalizer()
        all_test_vector = scaler.fit_transform(self.test_vector)

        scaler_patch = scaler.fit(self.patch_vector)
        all_patch_vector = scaler_patch.transform(self.patch_vector)

        dataset_test = np.delete(all_test_vector, failed_test_index, axis=0)
        dataset_patch = np.delete(all_patch_vector, failed_test_index, axis=0)
        dataset_name = np.delete(self.test_name, failed_test_index, axis=0)
        dataset_func = np.delete(self.test_data[3], failed_test_index, axis=0)

        patch_list = []
        smallest_dist = []
        for i in failed_test_index:
            failed_test_vector = all_test_vector[i]
            dists = []
            # find the k most closest test vector
            for j in range(len(dataset_test)):
                simi_test_vec = dataset_test[j]
                # dist = np.linalg.norm(simi_test_vec - failed_test_vector)
                dist = distance.euclidean(simi_test_vec, failed_test_vector)
                # dist = distance.cosine(simi_test_vec, failed_test_vector)
                dists.append([j, dist])
            k_index = sorted(dists, key=lambda x: float(x[1]), reverse=False)[:k]

            print('the smallest dist is {}'.format(k_index[0][1]))
            smallest_dist.append(k_index[0][1])

            if filt:
                # skip the test case with dist > 0.4
                k_index = np.array([v[0] for v in k_index if v[1] <= 0.5])
            else:
                k_index = np.array([v[0] for v in k_index])

            if k_index.size == 0:
                continue

            # check
            print('{}'.format(self.test_name[i]))
            # print('{}'.format(self.test_data[3][i]))
            print('similar test case:')
            k_simi_test = dataset_name[k_index]
            func = dataset_func[k_index]
            for t in range(len(k_simi_test)):
                print('{}'.format(k_simi_test[t]))
                # print('{}'.format(func[t]))
            print('--------------')

            k_patch_vector = dataset_patch[k_index]
            patch_list.append(k_patch_vector)
        return patch_list, scaler_patch, smallest_dist

    def get_patch_list4str(self, failed_test_index, k=10, filt=False):
        scaler = Normalizer()
        all_test_vector = scaler.fit_transform(self.test_vector)

        all_patch_vector = self.patch_vector

        dataset_test = np.delete(all_test_vector, failed_test_index, axis=0)
        dataset_patch = np.delete(all_patch_vector, failed_test_index, axis=0)
        dataset_name = np.delete(self.test_name, failed_test_index, axis=0)
        dataset_func = np.delete(self.test_data[3], failed_test_index, axis=0)

        patch_list = []
        smallest_dist = []
        for i in failed_test_index:
            failed_test_vector = all_test_vector[i]
            dists = []
            # find the k most closest test vector
            for j in range(len(dataset_test)):
                simi_test_vec = dataset_test[j]
                # dist = np.linalg.norm(simi_test_vec - failed_test_vector)
                dist = distance.euclidean(simi_test_vec, failed_test_vector)
                # dist = distance.cosine(simi_test_vec, failed_test_vector)
                dists.append([j, dist])
            k_index = sorted(dists, key=lambda x: float(x[1]), reverse=False)[:k]

            print('the smallest dist is {}'.format(k_index[0][1]))
            smallest_dist.append(k_index[0][1])

            if filt:
                # skip the test case with dist > 0.5
                k_index = np.array([v[0] for v in k_index if v[1] <= 0.5])
            else:
                k_index = np.array([v[0] for v in k_index])

            if k_index.size == 0:
                continue

            # check
            print('{}'.format(self.test_name[i]))
            print('{}'.format(self.test_data[3][i]))
            print('similar test case:')
            k_simi_test = dataset_name[k_index]
            func = dataset_func[k_index]
            for t in range(len(k_simi_test)):
                print('{}'.format(k_simi_test[t]))
                print('{}'.format(func[t]))
            print('--------------')

            k_patch_vector = dataset_patch[k_index]
            patch_list.append(k_patch_vector)
        return patch_list, None, smallest_dist

    def predict(self, patch_list, new_patch, scaler_patch):
        if self.patch_w2v != 'str':
            new_patch = scaler_patch.transform(new_patch.reshape((1, -1)))
        dist_final = []
        # patch list includes multiple patches for multi failed test cases
        for y in range(len(patch_list)):
            patches = patch_list[y]
            dist_k = []
            for z in range(len(patches)):
                vec = patches[z]
                # dist = np.linalg.norm(vec - new_patch)
                if self.patch_w2v == 'str':
                    dist = Levenshtein.distance(vec[0], new_patch[0])
                    dist_k.append(dist)
                else:
                    dist = distance.cosine(vec, new_patch)
                    # dist = distance.euclidean(vec, new_patch)/(1 + distance.euclidean(vec, new_patch))
                    dist_k.append(dist)

            dist_mean = np.array(dist_k).mean()
            dist_min = np.array(dist_k).min()

            # print('mean:{}  min:{}'.format(dist_mean, dist_min))
            dist_final.append(dist_mean)

        dist_final = np.array(dist_final).mean()
        return dist_final

    def predict_projects(self, path_collected_patch):
        projects = {'Chart': 26, 'Lang': 65, 'Time': 27, 'Closure': 176,  'Math': 106}
        # projects = {'Chart': 26}
        all_smallest_dist = []
        for project, number in projects.items():
            recommend_list_project = []
            print('Testing {}'.format(project))
            for id in range(1, number + 1):
                recommend_list = []
                print('{}_{} ------'.format(project, id))
                # extract failed test index according to bug_id
                project_id = '_'.join([project, str(id)])
                failed_test_index = [i for i in range(len(self.test_name)) if self.test_name[i].startswith(project_id+'-')]
                if failed_test_index == []:
                    print('No failed tests of this bugid found:{}'.format(project_id))
                    continue
                # find corresponding patches generated by tools
                # path_patch_sliced = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1ISSTA_sliced'
                available_path_patch = self.find_path_patch(path_collected_patch, project_id)
                if available_path_patch == []:
                    print('No tool patches found:{}'.format(project_id))
                    continue

                correct = incorrect = 0
                for p in available_path_patch:
                    if 'Correct' in p:
                        correct += 1
                    elif 'Incorrect' in p:
                        incorrect += 1
                # if correct == len(available_path_patch) or incorrect == len(available_path_patch):
                #     print('all same')
                #     continue

                # get patch list for failed test case
                if self.patch_w2v == 'str':
                    patch_list, scaler_patch, smallest_dist = self.get_patch_list4str(failed_test_index, k=5, filt=True)
                else:
                    patch_list, scaler_patch, smallest_dist = self.get_patch_list(failed_test_index, k=5, filt=True)
                all_smallest_dist += smallest_dist
                if patch_list == []:
                    print('no close test case found')
                    continue

                # return vector for path patch
                name_list, label_list, vector_list = self.vector4patch(available_path_patch)
                if not 0 in label_list or not 1 in label_list:
                    print('all same')
                    continue

                for i in range(len(name_list)):
                    name = name_list[i]
                    label = label_list[i]
                    vector_new_patch = vector_list[i]
                    dist = self.predict(patch_list, vector_new_patch, scaler_patch)
                    if self.patch_w2v == 'str':
                        score = 5000 - dist
                    else:
                        score = 1 - dist
                    if score == 0.0 or math.isnan(score):
                        continue
                    # record
                    recommend_list.append([name, label, score])
                    recommend_list_project.append([name, label, score])
                if recommend_list == []:
                    continue
                print('{} recommend list:'.format(project))
                recommend_list = pd.DataFrame(sorted(recommend_list, key=lambda x: x[2], reverse=True))
                Correct = recommend_list[recommend_list[1] == 1]
                Incorrect = recommend_list[recommend_list[1] == 0]
                plt.figure(figsize=(10, 4))
                plt.bar(Correct[:].index.tolist(), Correct[:][2], color="red")
                plt.bar(Incorrect[:].index.tolist(), Incorrect[:][2], color="lightgrey",)
                plt.xticks(recommend_list[:].index.tolist(), recommend_list[:][0].tolist())
                plt.xlabel('patchid by tool')
                plt.ylabel('Score of patch')
                plt.savefig('../fig/recommend_{}'.format(project_id))
                plt.cla()
                plt.close()
                # plt.show()
            print('{} recommend project:'.format(project))
            recommend_list_project = pd.DataFrame(sorted(recommend_list_project, key=lambda x: x[2], reverse=True))
            Correct = recommend_list_project[recommend_list_project[1] == 1]
            Incorrect = recommend_list_project[recommend_list_project[1] == 0]
            filter_out_incorrect = recommend_list_project.shape[0] - Correct[:].index.tolist()[-1] - 1
            print('{}: Incorrect filter rate: {}'.format(project, filter_out_incorrect/Incorrect.shape[0]))
            plt.bar(Correct[:].index.tolist(), Correct[:][2], color="red")
            plt.bar(Incorrect[:].index.tolist(), Incorrect[:][2], color="lightgrey")
            # plt.xticks(recommend_list_project[:].index.tolist(), recommend_list_project[:][0].tolist())
            plt.xlabel('patchid by tool')
            plt.ylabel('Score of patch')
            plt.title('recommend for {}'.format(project))
            plt.savefig('../fig/{}_recommend.png'.format(project))
            plt.cla()
            plt.close()
        plt.bar(range(len(all_smallest_dist)), all_smallest_dist,)

    def run(self):

        # load data and vector
        self.load_test()

        # pre-save bert vector for test dataset
        # patch_bert_vector.patch_bert()


        # validate hypothesis
        method_cluster = 'kmeans'
        number_cluster = 40
        clusters = self.cluster_test_dist(self.test_vector, method=method_cluster, number=number_cluster)
        self.patch_dist(self.patch_vector, clusters, method_cluster, number_cluster)


        # predict
        # self.predict_projects(self.path_collected_patch)


if __name__ == '__main__':
    config = Config()
    path_test = config.path_test
    path_patch_root = config.path_patch_root
    path_collected_patch = config.path_collected_patch

    path_test_function_patch_vector = config.path_test_function_patch_vector
    patch_w2v = config.patch_w2v

    e = Experiment(path_test, path_patch_root, path_collected_patch, path_test_function_patch_vector, patch_w2v)
    e.run()