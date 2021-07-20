import os
import csv

import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, confusion_matrix

patchsim_result = '/Users/haoye.tian/Documents/University/data/RESULT.csv'
path_patch = 'patch325.txt'
defects4j_V120_projects = ['Chart', 'Closure', 'Lang', 'Math', 'Mockito', 'Time']

def evaluation_metrics(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)

    print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(y_true), y_true.count(0), y_true.count(1)))
    print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall_p = tp / (tp + fn)
    recall_n = tn / (tn + fp)
    print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
    # return , auc_
    return auc_, recall_p, recall_n, acc, prc, rc, f1

def statistics(dict_result, path_patch):
    y_true, y_pred_prob, patch_names = [], [], []
    All, NoApplicable, Noresult = 0, 0, 0
    # for root, dirs, files in os.walk(path_patch):
    with open(path_patch, 'r+') as file:
        for line in file:
            line = line.strip()
            patch_name = '-'.join(line.split('-')[:4])
            label = int(line.split('-')[4])
            project = line.split('-')[1]
            id = line.split('-')[2]

            if project == 'Closure':
                continue

            All+= 1

            if project not in defects4j_V120_projects:
                NoApplicable += 1
                continue


            # search prediction result of patchsim
            if patch_name in dict_result.keys():
                if dict_result[patch_name] == 'Correct':
                    prediction = 1
                elif dict_result[patch_name] == 'Incorrect':
                    prediction = 0
                else:
                    prediction = -999
            else:
                prediction_list = []
                for i in range(10):
                    name_sliced_list = patch_name.split('-')
                    name_sliced_list[0] += '#' + str(i)
                    name_concatenated = '-'.join(name_sliced_list)
                    if name_concatenated in dict_result.keys():
                        if dict_result[name_concatenated] == 'Correct':
                            prediction_list.append(1)
                        elif dict_result[name_concatenated] == 'Incorrect':
                            prediction_list.append(0)
                        else:
                            pass
                if prediction_list != []:
                    prediction = np.array(prediction_list).mean()
                    # print(prediction_list)
                else:
                    prediction = -999

            # return numerical result
            if prediction == -999:
                Noresult += 1
                continue
            else:
                y_true.append(label)
                y_pred_prob.append(prediction)
                patch_names.append(patch_name)
    print('All: {}, NoApplicable: {}, NoResult: {}, Final: {}'.format(All, NoApplicable, Noresult, len(y_pred_prob)))
    return y_true, y_pred_prob, patch_names

def extractResult(patchsim_result):
    dict = {}
    times = []
    with open(patchsim_result, 'r+') as csvfile:
        rows = csv.reader(csvfile, delimiter=',', )
        for row in rows:
            # if isinstance(row, str):
            #     row = row.split(',')
            tool = row[0].split('_')[1]
            name = row[0].split('_')[0] + '-' + tool
            project = row[1]
            id = row[2]
            prediction = row[3]
            if len(row) == 5 and (prediction == 'Correct' or prediction == 'Incorrect'):
                time = row[4]
                times.append(float(time))

            dict[name] = prediction
    average_time = np.array(times).mean()
    print('average_time(second): {}'.format(average_time))
    return dict

def save_patchsim_result(y_true, y_pred_prob, patch_names, path_patch):
    lines = ''
    for i in range(len(patch_names)):
        lines += patch_names[i] + '-' + str(y_true[i]) + ',' + str(y_pred_prob[i]) + '\n'
    name = path_patch.split('.')[0]
    with open(name + '_result.txt' , 'w+') as f:
            f.write(lines)

if __name__ == '__main__':
    dict = extractResult(patchsim_result)
    y_true, y_pred_prob, patch_names = statistics(dict, path_patch)
    save_patchsim_result(y_true, y_pred_prob, patch_names, path_patch)
    evaluation_metrics(y_true, y_pred_prob)