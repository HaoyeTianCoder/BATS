import matplotlib.pyplot as plt
import numpy as np
import Levenshtein
import math
import pandas as pd
from sklearn.metrics import silhouette_score ,calinski_harabasz_score,davies_bouldin_score
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import os
from representation.word2vector import Word2vector
import json
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, average_precision_score
from mean_average_precision import MeanAveragePrecision

class evaluation:
    def __init__(self, patch_w2v, test_data, test_name, test_vector, patch_vector, exception_type):
        self.patch_w2v = patch_w2v

        self.test_data = test_data

        self.test_name = test_name
        # self.patch_name = None
        self.test_vector = test_vector
        self.patch_vector = patch_vector
        self.exception_type = exception_type

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
            if self.patch_w2v == 'bert':
                if os.path.exists(json_key):
                    with open(json_key, 'r+') as f:
                        vector_str = json.load(f)
                        vector = np.array(list(map(float, vector_str)))
                else:
                    w2v = Word2vector(patch_w2v='bert', )
                    vector = w2v.convert_single_patch(p)
                    vector_json = list(vector)
                    vector_json = list(map(str, vector_json))
                    with open(json_key, 'w+') as f:
                        jsonstr = json.dumps(vector_json, )
                        f.write(jsonstr)
            else:
                w2v = Word2vector(patch_w2v=self.patch_w2v, )
                vector = w2v.convert_single_patch(p)
            vector_list.append(vector)

        return name_list, label_list, vector_list

    def get_patch_list(self, failed_test_index, k=10, filt=0.7, model=None):
        scaler = Normalizer()
        all_test_vector = scaler.fit_transform(self.test_vector)

        scaler_patch = None
        if model == 'string':
            all_patch_vector = self.patch_vector
        else:
            scaler_patch = scaler.fit(self.patch_vector)
            all_patch_vector = scaler_patch.transform(self.patch_vector)

        dataset_test = np.delete(all_test_vector, failed_test_index, axis=0)
        dataset_patch = np.delete(all_patch_vector, failed_test_index, axis=0)
        dataset_name = np.delete(self.test_name, failed_test_index, axis=0)
        dataset_func = np.delete(self.test_data[3], failed_test_index, axis=0)
        dataset_exp = np.delete(self.exception_type, failed_test_index, axis=0)

        patch_list = []
        closest_score = []
        for i in failed_test_index:
            failed_test_vector = all_test_vector[i]
            # exception name of bug id
            exp_type = self.exception_type[i]
            if ':' in exp_type:
                exp_type = exp_type.split(':')[0]

            score_test = []
            # find the k most closest test vector
            for j in range(len(dataset_test)):
                simi_test_vec = dataset_test[j]
                # exception name of bug id
                simi_exp_type = dataset_exp[j]
                # if ':' in simi_exp_name:
                #     simi_exp_name = simi_exp_name.split(':')[0]

                dist = distance.euclidean(simi_test_vec, failed_test_vector) / (1 + distance.euclidean(simi_test_vec, failed_test_vector))
                # dist = distance.cosine(simi_test_vec, failed_test_vector)

                flag = 1 if exp_type == simi_exp_type else 0
                score_test.append([j, 1-dist, flag])
            k_index_list = sorted(score_test, key=lambda x: float(x[1]), reverse=True)[:k]
            closest_score.append(1-k_index_list[0][1])


            # keep the test case with simi score >= 0.7
            k_index = np.array([v[0] for v in k_index_list if v[1] >= filt])
            # k_index = np.array([v[0] for v in k_index_list])

            if k_index.size == 0:
                continue

            print('the closest test score is {}'.format(k_index_list[0][1]))

            # check
            print('{}'.format(self.test_name[i]))
            # print('{}'.format(self.test_data[3][i]))
            print('the similar test cases:')
            k_simi_test = dataset_name[k_index]
            func = dataset_func[k_index]
            for t in range(len(k_simi_test)):
                print('{}'.format(k_simi_test[t]))
                # print('{}'.format(func[t]))

            k_patch_vector = dataset_patch[k_index]
            patch_list.append(k_patch_vector)

            print('exception type: {}'.format(exp_type.split('.')[-1]))
            print('--------------')
        return patch_list, scaler_patch, closest_score

    def evaluate_collected_projects(self, path_collected_patch):
        projects = {'Chart': 26, 'Lang': 65, 'Math': 106, 'Time': 27}
        # projects = {'Math': 106}
        all_closest_score = []
        similarity_correct_minimum = 1
        similarity_incorrect = []
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
                    print('failed tests of this bugid not found:{}'.format(project_id))
                    continue
                # find corresponding patches generated by tools
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
                patch_list, scaler_patch, closest_score = self.get_patch_list(failed_test_index, k=1, filt=0.7, model=self.patch_w2v)
                all_closest_score += closest_score
                if patch_list == []:
                    print('no closest test case found')
                    continue

                # return vector for path patch
                name_list, label_list, vector_list = self.vector4patch(available_path_patch)
                # if not 0 in label_list or not 1 in label_list:
                #     print('all same')
                #     continue

                for i in range(len(name_list)):
                    name = name_list[i]
                    label = label_list[i]
                    vector_new_patch = vector_list[i]
                    dist = self.predict(patch_list, vector_new_patch, scaler_patch)
                    if self.patch_w2v == 'string':
                        score = 2800 - dist
                    else:
                        score = 1 - dist
                    if math.isnan(score):
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
                plt.savefig('../fig/RQ3/recommend_{}'.format(project_id))
                plt.cla()
                plt.close()
                # plt.show()

            # print('{} recommend project:'.format(project))
            if recommend_list_project == []:
                continue
            recommend_list_project = pd.DataFrame(sorted(recommend_list_project, key=lambda x: x[2], reverse=True))
            Correct = recommend_list_project[recommend_list_project[1] == 1]
            Incorrect = recommend_list_project[recommend_list_project[1] == 0]
            print('{}: {}'.format(project, recommend_list_project.shape[0]), end='  ')
            if Incorrect.shape[0] != 0 and Correct.shape[0] != 0:
                filter_out_incorrect = recommend_list_project.shape[0] - Correct[:].index.tolist()[-1] - 1
                print('Incorrect filter rate: {}'.format(filter_out_incorrect/Incorrect.shape[0]))
                # print('The minimum similarity score of the correct patch: {}'.format(np.array(Correct)[-1][2]))
                if np.array(Correct)[-1][2] < similarity_correct_minimum:
                    similarity_correct_minimum = np.array(Correct)[-1][2]
                similarity_incorrect.append(list(Incorrect[:][2]))
            plt.bar(Correct[:].index.tolist(), Correct[:][2], color="red")
            plt.bar(Incorrect[:].index.tolist(), Incorrect[:][2], color="lightgrey")
            # plt.xticks(recommend_list_project[:].index.tolist(), recommend_list_project[:][0].tolist())
            plt.xlabel('patchid by tool')
            plt.ylabel('Score of patch')
            plt.title('recommend for {}'.format(project))
            plt.savefig('../fig/RQ3/{}_recommend.png'.format(project))
            plt.cla()
            plt.close()
        print('The minimum similarity score of the correct patch: {}'.format(similarity_correct_minimum))
        for i in range(len(similarity_incorrect)):
            print('The number of incorrect patch: {}'.format(np.where(np.array(similarity_incorrect[i]) < similarity_correct_minimum)[0].size))
        plt.bar(range(len(all_closest_score)), sorted(all_closest_score, reverse=True),)
        plt.xlabel('the closest test case')
        plt.ylabel('Similarity Score of the closest test case')
        plt.title('Similarity of test case')
        plt.savefig('../fig/RQ3/Similarity_Test.png')

    def predict_collected_projects(self, path_collected_patch):
        projects = {'Chart': 26, 'Lang': 65, 'Math': 106, 'Time': 27}
        # projects = {'Math': 106}
        all_closest_score = []
        y_preds = []
        y_trues = []
        similarity_correct_minimum = 1
        similarity_incorrect = []
        for project, number in projects.items():
            recommend_list_project = []
            print('Testing {}'.format(project))
            for id in range(1, number + 1):
                print('{}_{} ------'.format(project, id))
                # extract failed test index according to bug_id
                project_id = '_'.join([project, str(id)])
                failed_test_index = [i for i in range(len(self.test_name)) if self.test_name[i].startswith(project_id+'-')]
                if failed_test_index == []:
                    print('failed tests of this bugid not found:{}'.format(project_id))
                    continue
                # find corresponding patches generated by tools
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
                patch_list, scaler_patch, closest_score = self.get_patch_list(failed_test_index, k=5, filt=0.7, model=self.patch_w2v)
                all_closest_score += closest_score
                if patch_list == []:
                    print('no closest test case found')
                    continue

                # return vector for path patch
                name_list, label_list, vector_list = self.vector4patch(available_path_patch)
                # if not 0 in label_list or not 1 in label_list:
                #     print('all same')
                #     continue

                recommend_list = []
                # centers, threshold_list = self.dynamic_threshold(patch_list)
                centers, threshold_list = self.dynamic_threshold2(patch_list, sumup='mean')
                for i in range(len(name_list)):
                    name = name_list[i]
                    vector_new_patch = vector_list[i]
                    y_true = label_list[i]
                    # y_pred = self.predict_label(centers, threshold_list, vector_new_patch, scaler_patch)
                    # y_pred_prob = self.predict_prob(centers, threshold_list, vector_new_patch, scaler_patch)
                    y_pred_prob, y_pred = self.predict_recom(centers, threshold_list, vector_new_patch, scaler_patch)

                    recommend_list.append([name, y_pred, y_true, y_pred_prob])

                    y_preds.append(y_pred)
                    y_trues.append(y_true)

                self.evaluate_recommend(recommend_list)


        recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_trues, y_preds)

    def predict(self, patch_list, new_patch, scaler_patch):
        if self.patch_w2v != 'string':
            new_patch = scaler_patch.transform(new_patch.reshape((1, -1)))
        dist_final = []
        # patch list includes multiple patches for multi failed test cases
        for y in range(len(patch_list)):
            patches = patch_list[y]
            dist_k = []
            for z in range(len(patches)):
                vec = patches[z]
                # dist = np.linalg.norm(vec - new_patch)
                if self.patch_w2v == 'string':
                    dist = Levenshtein.distance(vec[0], new_patch[0])
                    dist_k.append(dist)
                else:
                    # choose method to calculate distance
                    dist = distance.cosine(vec, new_patch)
                    # dist = distance.euclidean(vec, new_patch)/(1 + distance.euclidean(vec, new_patch))
                    dist_k.append(dist)

            dist_mean = np.array(dist_k).mean()
            dist_min = np.array(dist_k).min()

            # print('mean:{}  min:{}'.format(dist_mean, dist_min))
            dist_final.append(dist_min)

        dist_final = np.array(dist_final).mean()
        return dist_final

    def dynamic_threshold(self, patch_list):
        centers = []
        threshold_list = []
        # patch list includes multiple patches for multi failed test cases
        for y in range(len(patch_list)):
            patches = patch_list[y]

            # threshold 1: center of patch list
            center = np.array(patches).mean(axis=0)
            dist_mean = np.array([distance.cosine(p, center) for p in patches]).mean()
            # dist_mean = np.array([distance.cosine(p, center) for p in patches]).max()
            score_mean = 1-dist_mean

            centers.append(center)
            threshold_list.append(score_mean)
        return centers, threshold_list

    def dynamic_threshold2(self, patch_list, sumup='mean'):
        centers = []
        threshold_list = []
        if len(patch_list) == 1:
            center = patch_list[0].mean(axis=0)
            if sumup == 'mean':
                dist_mean = np.array([distance.cosine(p, center) for p in patch_list[0]]).mean()
            elif sumup == 'max':
                pass
                # dist_mean = np.array([distance.cosine(p, center) for p in patch_list[0]]).max()
        else:
            patches = patch_list[0]
            for i in range(1, len(patch_list)):
                patches = np.concatenate((patches, patch_list[i]), axis=0)
            center = patches.mean(axis=0)
            if sumup == 'mean':
                dist_mean = np.array([distance.cosine(p, center) for p in patches]).mean()
            elif sumup == 'max':
                dist_mean = np.array([distance.cosine(p, center) for p in patches]).max()

        score_mean = 1 - dist_mean

        return [center], [score_mean]


    def predict_label(self, centers, threshold_list, new_patch, scaler_patch):
        if self.patch_w2v != 'string':
            new_patch = scaler_patch.transform(new_patch.reshape((1, -1)))

        vote_list = []
        # patch list includes multiple patches for multi failed test cases
        for y in range(len(centers)):
            center = centers[y]
            score_mean = threshold_list[y]

            # choose method to calculate distance
            dist_new = distance.cosine(new_patch, center)
            # dist_new = distance.euclidean(vec, new_patch)/(1 + distance.euclidean(vec, new_patch))

            score_new = 1 - dist_new

            vote_list.append(1 if score_new >= score_mean else 0)
        if vote_list.count(1) >= len(centers) / 2.0:
            return 1
        else:
            return 0

    def predict_prob(self, centers, threshold_list, new_patch, scaler_patch):
        if self.patch_w2v != 'string':
            new_patch = scaler_patch.transform(new_patch.reshape((1, -1)))

        center = centers[0]
        score_mean = threshold_list[0]

        # choose method to calculate distance
        # dist_new = distance.cosine(new_patch, center)
        dist_new = distance.euclidean(new_patch, center)/(1 + distance.euclidean(new_patch, center))

        score_prob_new = 1 - dist_new

        return score_prob_new

    def predict_recom(self, centers, threshold_list, new_patch, scaler_patch):
        if self.patch_w2v != 'string':
            new_patch = scaler_patch.transform(new_patch.reshape((1, -1)))

        center = centers[0]
        score_mean = threshold_list[0]

        # choose method to calculate distance
        # dist_new = distance.cosine(new_patch, center)
        dist_new = distance.euclidean(new_patch, center)/(1 + distance.euclidean(new_patch, center))

        score_prob_new = 1 - dist_new
        if score_prob_new >= score_mean:
            y_pred = 1
        else:
            y_pred = 0

        return score_prob_new, y_pred

    def evaluation_metrics(self, y_trues, y_pred_probs):
        fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_pred_probs, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_preds = [1 if p >= 0.5 else 0 for p in y_pred_probs]

        acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
        prc = precision_score(y_true=y_trues, y_pred=y_preds)
        rc = recall_score(y_true=y_trues, y_pred=y_preds)
        f1 = 2 * prc * rc / (prc + rc)

        print('------------')
        print('The number of dataset: {}, Correct: {}, Incorrect: {}'.format(len(y_trues), y_trues.count(1), y_trues.count(0)))
        print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
        # return , auc_

        print('AP: {}'.format(average_precision_score(y_trues, y_pred_probs)))
        return recall_p, recall_n, acc, prc, rc, f1

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

    def evaluate_recommend(self, recommend_list):
        recommend_list = pd.DataFrame(sorted(recommend_list, key=lambda x: x[3], reverse=True))
        recommend_list = recommend_list[recommend_list[1] == 1]
        k = 5
        number_correct = 0
        precision_all = 0
        if recommend_list.shape[0] >= k:
            for i in range(recommend_list.shape[0]):
                if recommend_list[i][2] == 1:
                    number_correct += 1.0
                    precision_all += number_correct / (i + 1)
        AP = precision_all / number_correct
        print('AP@{} for recommend list: {}'.format(k, AP))
        RR = 1.0 / (list(recommend_list[:][2]).index(1) + 1)
        print('RR for recommend list: {}'.format(RR))