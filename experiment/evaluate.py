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
from experiment.ML4prediction import MlPrediction
from tqdm import tqdm
import seaborn as sns
from matplotlib.patches import PathPatch
import scipy.stats as stats

notRecognizedByBert = ['Correct-Lang-6-patch1', 'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1', 'Correct-Lang-25-patch1', 'Correct-Lang-53-patch1', 'Incorrect-Math-6-patch2', 'Incorrect-Math-6-patch2', 'Incorrect-Math-6-patch1', 'Correct-Math-56-patch1', 'Incorrect-Math-80-patch1', 'Incorrect-Math-104-patch1']
notRecognizedByCC2Vec = ['Correct-Lang-25-patch1', 'Correct-Lang-53-patch1', 'Correct-Math-56-patch1', 'Incorrect-Math-80-patch1']
notRecognized = notRecognizedByBert + notRecognizedByCC2Vec

class evaluation:
    def __init__(self, patch_w2v, test_data, test_name, test_vector, patch_vector, exception_type):
        self.patch_w2v = patch_w2v

        self.test_data = test_data

        self.test_name = test_name
        # self.patch_name = None
        self.test_vector = test_vector
        self.patch_vector = patch_vector
        self.exception_type = exception_type

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

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

    def vector4patch(self, available_path_patch, compare='ASE2020',):
        vector_list = []
        vector_ML_list = []
        label_list = []
        name_list = []
        for p in available_path_patch:
            recogName = '-'.join([p.split('/')[-4], p.split('/')[-3], p.split('/')[-2], p.split('/')[-1]])
            if recogName in notRecognized: # some specific patches can not be recognized
                continue

            # vector
            json_key = p + '_.json' # pre-saved bert vector
            json_key_cross = p + '_cross.json' # pre-saved bert feature in Haoye's ASE2020
            if self.patch_w2v == 'bert':
                if os.path.exists(json_key):
                    with open(json_key, 'r+') as f:
                        vector_str = json.load(f)
                        vector = np.array(list(map(float, vector_str)))
                else:
                    w2v = Word2vector(patch_w2v='bert', )
                    vector, vector_ML = w2v.convert_single_patch(p)
                    vector_json = list(map(str, list(vector)))
                    vector_json_cross = list(map(str, list(vector)))
                    with open(json_key, 'w+') as f:
                        jsonstr = json.dumps(vector_json, )
                        f.write(jsonstr)
                    # with open(json_key_cross, 'w+') as f:
                    #     jsonstr = json.dumps(vector_json_cross, )
                    #     f.write(jsonstr)
            elif self.patch_w2v == 'cc2vec':
                w2v = Word2vector(patch_w2v=self.patch_w2v, )
                vector, _ = w2v.convert_single_patch(p)
            elif self.patch_w2v == 'string':
                w2v = Word2vector(patch_w2v=self.patch_w2v, )
                vector, _ = w2v.convert_single_patch(p)
            else:
                raise
            # if list(vector.astype(float)) == list(np.zeros(240).astype(float)) or list(vector.astype(float)) == list(np.zeros(1024).astype(float)):
            #     ttt = '-'.join([p.split('/')[-4], p.split('/')[-3], p.split('/')[-2], p.split('/')[-1]])
            #     notRecognized.append(ttt)
            vector_list.append(vector)

            # compared with Haoye's ASE2020
            if compare=='ASE2020':
                with open(json_key_cross, 'r+') as f:
                    vector_str = json.load(f)
                    vector_ML = np.array(list(map(float, vector_str)))
                vector_ML_list.append(vector_ML)

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

        return name_list, label_list, vector_list, vector_ML_list

    def vector4patch_patchsim(self, available_path_patch, compare='ASE2020',):
        vector_list = []
        vector_ML_list = []
        label_list = []
        name_list = []
        for p in available_path_patch:

            # vector
            json_key = p + '_.json'
            json_key_cross = p + '_cross.json'
            if self.patch_w2v == 'bert':
                if os.path.exists(json_key):
                    with open(json_key, 'r+') as f:
                        vector_str = json.load(f)
                        vector = np.array(list(map(float, vector_str)))
                else:
                    w2v = Word2vector(patch_w2v='bert', )
                    vector, vector_ML = w2v.convert_single_patch(p)
                    vector_json = list(map(str, list(vector)))
                    vector_json_cross = list(map(str, list(vector)))
                    with open(json_key, 'w+') as f:
                        jsonstr = json.dumps(vector_json, )
                        f.write(jsonstr)
                    # with open(json_key_cross, 'w+') as f:
                    #     jsonstr = json.dumps(vector_json_cross, )
                    #     f.write(jsonstr)
            elif self.patch_w2v == 'cc2vec':
                w2v = Word2vector(patch_w2v=self.patch_w2v, )
                vector, _ = w2v.convert_single_patch(p)
            elif self.patch_w2v == 'string':
                w2v = Word2vector(patch_w2v=self.patch_w2v, )
                vector, _ = w2v.convert_single_patch(p)
            else:
                raise
            # if list(vector.astype(float)) == list(np.zeros(240).astype(float)) or list(vector.astype(float)) == list(np.zeros(1024).astype(float)):
            #     ttt = '-'.join([p.split('/')[-4], p.split('/')[-3], p.split('/')[-2], p.split('/')[-1]])
            #     notRecognized.append(ttt)
            vector_list.append(vector)
            if compare=='ASE2020':
                with open(json_key_cross, 'r+') as f:
                    vector_str = json.load(f)
                    vector_ML = np.array(list(map(float, vector_str)))
                vector_ML_list.append(vector_ML)

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

        return name_list, label_list, vector_list, vector_ML_list

    def get_associated_patch_list(self, failed_test_index, k=5, cut_off=0.0, model=None):
        scaler = Normalizer()
        all_test_vector = scaler.fit_transform(self.test_vector)

        scaler_patch = None
        if model == 'string':
            all_patch_vector = self.patch_vector
        else:
            scaler_patch = scaler.fit(self.patch_vector)
            all_patch_vector = scaler_patch.transform(self.patch_vector)

        # construct new test and patch dataset(repository) by excluding the current failed test cases being predicted
        dataset_test = np.delete(all_test_vector, failed_test_index, axis=0)
        dataset_patch = np.delete(all_patch_vector, failed_test_index, axis=0)
        dataset_name = np.delete(self.test_name, failed_test_index, axis=0)
        dataset_func = np.delete(self.test_data[3], failed_test_index, axis=0)
        dataset_exp = np.delete(self.exception_type, failed_test_index, axis=0)

        patch_list = [] # the associated patches of similar test cases
        closest_score = []
        for i in failed_test_index:
            failed_test_vector = all_test_vector[i]

            # Deprecated. exception type of current bug id.
            exp_type = self.exception_type[i]
            if ':' in exp_type:
                exp_type = exp_type.split(':')[0]

            score_test = []
            # find the k most closest test vector from other bug-id
            for j in range(len(dataset_test)):
                simi_test_vec = dataset_test[j]

                # Deprecated. exception type from other bug-id.
                simi_exp_type = dataset_exp[j]
                if ':' in simi_exp_type:
                    simi_exp_type = simi_exp_type.split(':')[0]
                flag = 1 if exp_type == simi_exp_type else 0

                dist = distance.euclidean(simi_test_vec, failed_test_vector) / (1 + distance.euclidean(simi_test_vec, failed_test_vector))
                score_test.append([j, 1-dist, flag]) # we use similarity instead of distance
            k_index_list = sorted(score_test, key=lambda x: float(x[1]), reverse=True)[:k]
            closest_score.append(k_index_list[0][1])
            # print('the closest test score is {}'.format(k_index_list[0][1]))

            # keep the test case with simi score >= 0.8 or *
            k_index = np.array([v[0] for v in k_index_list if v[1] >= cut_off])

            if k_index.size == 0:
                continue

            # exhibit the similar test case
            print('******')
            print('{}'.format(self.test_name[i]))
            print('the similar test cases:')
            k_simi_test = dataset_name[k_index]
            func = dataset_func[k_index]
            for t in range(len(k_simi_test)):
                print('{}'.format(k_simi_test[t]))
                # print('{}'.format(func[t]))

            k_patch_vector = dataset_patch[k_index]
            patch_list.append(k_patch_vector)

            # print('exception type: {}'.format(exp_type.split('.')[-1]))
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

                # get patch list for failed test case
                patch_list, scaler_patch, closest_score = self.get_patch_list(failed_test_index, k=1, cut_off=0.7, model=self.patch_w2v)
                all_closest_score += closest_score
                if patch_list == []:
                    print('no closest test case found')
                    continue

                # return vector for path patch
                name_list, label_list, vector_list, vector_ML_list = self.vector4patch(available_path_patch, 'False')
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

    def predict_collected_projects(self, path_collected_patch=None, cut_off=0.8, distance_method = distance.cosine, patchsim=False):
        print('Research Question 3')
        projects = {'Chart': 26, 'Lang': 65, 'Math': 106, 'Time': 27}
        y_preds, y_trues = [], []
        MAP, MRR, number_patch_MAP = [], [], 0
        recommend_list_project = []
        x_train, y_train, x_test, y_test = [], [], [], []
        comparison = 'No2020' # will make comparison if the value equals to 'ASE2020'
        box_projecs_co, box_projecs_inco, projects_name = [], [], []
        mean_stand_dict = {0.0: [443, 816], 0.6: [273, 246], 0.7: [231, 273], 0.8: [180, 235], 0.9: [130, 130]}
        print('test case similarity cut-off: {}'.format(cut_off))
        for project, number in projects.items():
            print('Testing {}'.format(project))
            for id in range(1, number + 1):
                print('----------------')
                print('{}_{}'.format(project, id))
                project_id = '_'.join([project, str(id)])

                # extract failed test index according to bug_id
                failed_test_index = [i for i in range(len(self.test_name)) if self.test_name[i].startswith(project_id+'-')]
                if failed_test_index == []:
                    print('Couldnt find any failed test case for this bugid: {}'.format(project_id))
                    # print('{} patches skipped'.format(len(available_path_patch)))
                    continue

                # find paths of patches generated by tools
                available_path_patch = self.find_path_patch(path_collected_patch, project_id)
                if available_path_patch == []:
                    print('No generated patches of APR tools found:{}'.format(project_id))
                    continue

                # return vector according to available_path_patch
                if patchsim:
                    name_list, label_list, generated_patch_list, vector_ML_list = self.vector4patch_patchsim(available_path_patch, compare=comparison,)
                else:
                    name_list, label_list, generated_patch_list, vector_ML_list = self.vector4patch(available_path_patch, compare=comparison,)
                if name_list == []:
                    print('all the patches can not be recognized')
                    continue
                # plot distribution of correct and incorrect patches
                co = label_list.count(1)
                inco = label_list.count(0)
                box_projecs_co.append(co)
                box_projecs_inco.append(inco)
                projects_name.append(project)

                # access the associated patch list(patch repository) of similar failed test cases
                associated_patch_list, scaler_patch, closest_score = self.get_associated_patch_list(failed_test_index, k=5, cut_off=cut_off, model=self.patch_w2v)
                if associated_patch_list == []:
                    print('No closest test case that satisfied with the condition of cut-off similarity')
                    print('save train data for ML model of ASE2020')
                    # comparison with ML prediction in ASE2020
                    if comparison == 'ASE2020' and vector_ML_list != []:
                        for i in range(len(label_list)):
                            # if list(vector_list[i].astype(float)) != list(np.zeros(240).astype(float)):
                                x_train.append(vector_ML_list[i])
                                y_train.append(label_list[i])
                    continue

                recommend_list = []
                # calculate the center of associated patches(repository)
                centers = self.dynamic_threshold2(associated_patch_list, distance_method=distance_method, sumup='mean')
                for i in range(len(name_list)):
                    name = name_list[i]
                    tested_patch = generated_patch_list[i]
                    y_true = label_list[i]
                    # y_pred = self.predict_label(centers, threshold_list, vector_new_patch, scaler_patch)
                    # y_pred_prob = self.predict_prob(centers, threshold_list, vector_new_patch, scaler_patch)
                    y_pred_prob, y_pred = self.predict_recom(centers, tested_patch, scaler_patch, mean_stand_dict[cut_off], distance_method=distance_method,)

                    if not math.isnan(y_pred_prob):
                        recommend_list.append([name, y_pred, y_true, y_pred_prob])

                        y_preds.append(y_pred_prob)
                        y_trues.append(y_true)

                        # ML prediction for comparison
                        if comparison == 'ASE2020':
                            x_test.append(vector_ML_list[i])
                            y_test.append(y_true)


                if not (not 1 in label_list or not 0 in label_list) and recommend_list != []: # ensure there are correct and incorrect patches in recommended list
                    AP, RR = self.evaluate_recommend_list(recommend_list)
                    if AP != None and RR != None:
                        MAP.append(AP)
                        MRR.append(RR)
                        number_patch_MAP += len(recommend_list)

                recommend_list_project += recommend_list

        # evaluation based on a few metrics
        self.evaluation_metrics(y_trues, y_preds)

        self.MAP_MRR_Mean(MAP, MRR, number_patch_MAP)

        if comparison == 'ASE2020' and cut_off > 0.0:
            print('------')
            print('Evaluating ASE2020 Performance')
            MlPrediction(x_train, y_train, x_test, y_test, algorithm='lr').predict()
            MlPrediction(x_train, y_train, x_test, y_test, algorithm='rf').predict()

        if patchsim:
            print('------')
            print('Evaluating Incorrect Excluded on PatchSim')
            # [name, y_pred, y_true, y_pred_prob]
            recommend_list_project = pd.DataFrame(sorted(recommend_list_project, key=lambda x: x[3], reverse=True))
            Correct = recommend_list_project[recommend_list_project[2]==1]
            filter_out_incorrect = recommend_list_project.shape[0] - Correct[:].index.tolist()[-1] - 1
            print('Exclude incorrect: {}'.format(filter_out_incorrect))
            print('Exclude rate: {}'.format(filter_out_incorrect/recommend_list_project.shape[0]))

        # self.statistics_box(box_projecs_co, box_projecs_inco, projects_name)

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

    def dynamic_threshold2(self, patch_list, distance_method=distance.euclidean, sumup='mean'):
        # patch_list: [[top-5 patches for failed test case 1], [top-5 patches failed test case 2], [top-5 patches failed test case 3]]
        if self.patch_w2v != 'string':
            if len(patch_list) == 1:
                center = patch_list[0].mean(axis=0)
                # if sumup == 'mean':
                #     dist_mean = np.array([distance_method(p, center) for p in patch_list[0]]).mean()
                # elif sumup == 'max':
                #     dist_mean = np.array([distance_method(p, center) for p in patch_list[0]]).max()
            else:
                # calculate center
                patches = patch_list[0]
                for i in range(1, len(patch_list)):
                    patches = np.concatenate((patches, patch_list[i]), axis=0)
                center = patches.mean(axis=0)
                # if sumup == 'mean':
                #     dist_mean = np.array([distance_method(p, center) for p in patches]).mean()
                # elif sumup == 'max':
                #     dist_mean = np.array([distance_method(p, center) for p in patches]).max()
        else:
            return patch_list

        return [center]


    def predict_label(self, centers, threshold_list, new_patch, scaler_patch, ):
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

    def predict_prob(self, centers, threshold_list, new_patch, scaler_patch, distance_method=distance.euclidean):
        if self.patch_w2v != 'string':
            new_patch = scaler_patch.transform(new_patch.reshape((1, -1)))

        center = centers[0]

        dist_new = distance_method(new_patch, center)

        # normalize range
        if distance_method == distance.euclidean:
            dist_new = dist_new / (1+dist_new)
            score_prob_new = 1 - dist_new

        elif distance_method == distance.cosine:
            dist_new = dist_new / (1 + dist_new)
            score_prob_new = 1 - dist_new

        return score_prob_new

    def predict_recom(self, centers, new_patch, scaler_patch, mean_stand, distance_method=distance.euclidean):
        if self.patch_w2v != 'string':
            new_patch = scaler_patch.transform(new_patch.reshape((1, -1)))

            center = centers[0]
            dist_new = distance_method(new_patch, center)

            # normalize range
            # score_prob_new = self.sigmoid(1 - dist_new)
            dist_new = dist_new / (1+dist_new)
            score_prob_new = 1 - dist_new

            # if score_prob_new >= score_mean:
            if score_prob_new >= 0.5:
                y_pred = 1
            else:
                y_pred = 0
            return score_prob_new, y_pred

        else:
            new_patch = new_patch[0]
            dist_new = []
            # mean distance to every patch
            for i in range(len(centers)):
                patches_top5 = centers[i]
                for p in patches_top5:
                    dist_new.append(Levenshtein.distance(new_patch, str(p)))
            dist_new = np.array(dist_new).mean()

            # (dist_new-mean)/stand
            dist_new = (dist_new-mean_stand[0])/mean_stand[1]
            try:
                score_prob_new = self.sigmoid(-dist_new)
            except:
                print(dist_new)

            if score_prob_new >= 0.5:
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

        print('\n***------------***')
        print('Evaluating AUC, F1, +Recall, -Recall')
        print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(y_trues), y_trues.count(0), y_trues.count(1)))
        print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
        # return , auc_

        # print('AP: {}'.format(average_precision_score(y_trues, y_pred_probs)))
        return recall_p, recall_n, acc, prc, rc, f1

    def evaluate_defects4j_projects(self, ):
        print('Research Question 2')
        scaler = Normalizer()
        all_test_vector = scaler.fit_transform(self.test_vector)
        scaler_patch = scaler.fit(self.patch_vector)
        all_patch_vector = scaler_patch.transform(self.patch_vector)

        projects = {'Chart': 26, 'Lang': 65, 'Time': 27, 'Closure': 176, 'Math': 106, 'Cli': 40, 'Codec': 18, 'Compress': 47, 'Collections': 28,  'JacksonCore': 26, 'JacksonDatabind': 112, 'JacksonXml': 6, 'Jsoup': 93, 'Csv': 16, 'Gson': 18, 'JxPath': 22, 'Mockito': 38}
        # projects = {'Chart': 26, 'Lang': 65, 'Time': 27, 'Math': 106, }
        all_closest_score = []
        box_plot = []
        for project, number in projects.items():
            print('Testing {}'.format(project))

            # go through all test cases
            cnt = 0
            for i in range(len(self.test_name)):
                # skip other projects while testing one project
                if not self.test_name[i].startswith(project):
                    continue
                # project = self.test_name[i].split('-')[0].split('_')[0]
                id = self.test_name[i].split('-')[0].split('_')[1]
                print('{}'.format(self.test_name[i]))
                this_test = all_test_vector[i]
                this_patch = all_patch_vector[i]

                # find the closest test case
                dist_min_index = None
                dist_min = np.inf
                for j in range(len(all_test_vector)):
                    # skip itself
                    if j == i:
                        continue
                    # whether skip current project-id
                    if self.test_name[j].startswith(project+'_'+id+'-'):
                        continue
                    dist = distance.euclidean(this_test, all_test_vector[j])/(1 + distance.euclidean(this_test, all_test_vector[j]))
                    if dist < dist_min:
                        dist_min = dist
                        dist_min_index = j
                sim_test = 1 - dist_min
                all_closest_score.append(sim_test)
                if sim_test >= 0.6:
                    # find associated patches similarity
                    print('the closest test: {}'.format(self.test_name[dist_min_index]))
                    closest_patch = all_patch_vector[dist_min_index]
                    distance_patch = distance.euclidean(closest_patch, this_patch)/(1 + distance.euclidean(closest_patch, this_patch))
                    score_patch = 1 - distance_patch
                    if math.isnan(score_patch):
                        continue
                    box_plot.append([project, 'H', score_patch])

                # find average patch similarity
                simi_patch_average = []
                for p in range(len(all_patch_vector)):
                    if p == i:
                        continue
                    dist = distance.euclidean(this_patch, all_patch_vector[p]) / (1 + distance.euclidean(this_patch, all_patch_vector[p]))
                    simi_patch = 1 - dist
                    if math.isnan(simi_patch):
                        continue
                    simi_patch_average.append(simi_patch)
                box_plot.append([project, 'N', np.array(simi_patch_average).mean()])

                # project_list.append([self.test_name[i], score_patch])


            # if project_list == []:
            #     print('{} no found'.format(project))
            #     continue
            # recommend_list_project = pd.DataFrame(sorted(project_list, key=lambda x: x[1], reverse=True))
            # plt.bar(recommend_list_project.index.tolist(), recommend_list_project[:][1], color='chocolate')
            # # plt.bar(recommend_list_project.index.tolist(), recommend_list_project[:][1], color='steelblue')
            # plt.xlabel('Failed test cases', fontsize=14)
            # plt.ylabel('Similarity of the associated patches', fontsize=14)
            # # plt.title('Similarity distribution of {}'.format(project))
            # plt.savefig('../fig/RQ2/distance_patch_{}'.format(project))
            # plt.cla()


        plt.figure(figsize=(10, 5))
        plt.xticks(fontsize=15, )
        plt.yticks(fontsize=15, )
        plt.bar(range(len(all_closest_score)), sorted(all_closest_score, reverse=True),)
        plt.xlabel('ID of failing test cases', fontsize=20)
        plt.ylabel('Similarity score \nwith the closest test cases', fontsize=20)
        # plt.title('Similarity of test case')
        plt.savefig('../fig/RQ2/Similarity_Test.png')
        plt.close()


        dfl = pd.DataFrame(box_plot)
        dfl.columns = ['Project', 'Label', 'Similarity of patch']
        # b, c = dfl.iloc[0].copy(), dfl.iloc[15].copy()
        # dfl.iloc[0], dfl.iloc[15] = c, b
        colors = {'H': 'white', 'N': 'grey'}
        fig = plt.figure(figsize=(15, 8))
        plt.xticks(fontsize=28, )
        plt.yticks(fontsize=28, )
        bp = sns.boxplot(x='Project', y='Similarity of patch', data=dfl, showfliers=False, palette=colors, hue='Label', width=0.7, )
        bp.set_xticklabels(bp.get_xticklabels(), rotation=320)
        # bp.set_xticklabels(bp.get_xticklabels(), fontsize=28)
        # bp.set_yticklabels(bp.get_yticklabels(), fontsize=28)
        plt.xlabel('Project', size=31)
        plt.ylabel('Similarity score', size=30)
        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                   borderaxespad=0, ncol=3, fontsize=30, )
        self.adjust_box_widths(fig, 0.8)
        plt.tight_layout()
        # plt.show()
        plt.savefig('../fig/RQ2/boxplot.png')

        H_stat = dfl[dfl['Label'] == 'H'].iloc[:, 2].tolist()
        N_stat = dfl[dfl['Label'] == 'N'].iloc[:, 2].tolist()
        hypo = stats.mannwhitneyu(H_stat, N_stat, alternative='two-sided')
        print(hypo)


    def evaluate_recommend_list(self, recommend_list):
        # recommend_list: [name, y_pred, y_true, y_pred_prob]
        recommend_list = pd.DataFrame(sorted(recommend_list, key=lambda x: x[3], reverse=True)) # rank by prediction probability
        number_correct = 0.0
        precision_all = 0.0

        for i in range(recommend_list.shape[0]):
            if recommend_list.loc[i][2] == 1:
                number_correct += 1.0
                precision_all += (number_correct / (i + 1))

        if number_correct == 0.0:
            print('No correct patch found on the recommended list')
            return None, None
        else:
            AP = precision_all / number_correct
            RR = 1.0 / (list(recommend_list[:][2]).index(1) + 1)

        print('AP: {}'.format(AP))
        print('RR: {}'.format(RR))

        return AP, RR

    def MAP_MRR_Mean(self, MAP, MRR, number_patch_MAP):
        print('------')
        print('Evaluating MAP, MRR on Recommended List')
        print('Patch size: {}'.format(number_patch_MAP))
        print('Bug project size: {}'.format(len(MAP)))
        print('MAP: {}, MRR: {}'.format(np.array(MAP).mean(), np.array(MRR).mean()))

    def statistics_box(self, box_projecs_co, box_projecs_inco, projects_name):
        data = {
            'Correct': box_projecs_co,
            'Incorrect': box_projecs_inco,
            'Project': projects_name
        }
        df = pd.DataFrame(data)
        dfl = pd.melt(df, id_vars='Project', value_vars=['Correct', 'Incorrect'], )
        dfl.columns = ['Project', 'Label', 'Number of Patches']
        colors = {'Correct': 'white', 'Incorrect': 'darkgrey'}

        fig = plt.figure(figsize=(10, 5))
        plt.xticks(fontsize=15, )
        plt.yticks(fontsize=15, )
        plt.legend(fontsize=15)
        bp = sns.boxplot(x='Project', y='Number of Patches', data=dfl, showfliers=False, palette=colors, hue='Label', width=0.6, )
        plt.xlabel('Project', fontsize=17)
        plt.ylabel('Number of Patches', fontsize=17)
        self.adjust_box_widths(fig, 0.8)

        plt.show()

    def adjust_box_widths(self, g, fac):
        """
        Adjust the widths of a seaborn-generated boxplot.
        """

        # iterating through Axes instances
        for ax in g.axes:
            # iterating through axes artists:
            for c in ax.get_children():

                # searching for PathPatches
                if isinstance(c, PathPatch):
                    # getting current width of box:
                    p = c.get_path()
                    verts = p.vertices
                    verts_sub = verts[:-1]
                    xmin = np.min(verts_sub[:, 0])
                    xmax = np.max(verts_sub[:, 0])
                    xmid = 0.5 * (xmin + xmax)
                    xhalf = 0.5 * (xmax - xmin)

                    # setting new width of box
                    xmin_new = xmid - fac * xhalf
                    xmax_new = xmid + fac * xhalf
                    verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                    verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                    # setting new width of median line
                    for l in ax.lines:
                        if np.all(l.get_xdata() == [xmin, xmax]):
                            l.set_xdata([xmin_new, xmax_new])