import json
import pickle
from collect.code_processing_all import get_all
from collect.configV2 import *


def get_data_json(path):
    file = open(path, 'r')
    dic = json.load(file)
    return dic


def get_link_patch():

    four_list = get_all(PATH_PROJECTS, NAME_LIST)
    dic_haoye = get_data_json(path_haoye)
    dic_yinghua = get_data_json(path_yinghua)
    dic_weiguo = get_data_json(path_weiguo)
    dic_all = {}
    dic_all.update(dic_haoye)
    dic_all.update(dic_yinghua)
    dic_all.update(dic_weiguo)

    name_number_func_list = []
    error_title_list = []
    error_message_list = []
    func_message_list = []
    patch_list = []

    for ind in range(len(four_list[0])):
        name_number_func_list.append(four_list[0][ind])
        error_title_list.append(four_list[1][ind])
        error_message_list.append(four_list[2][ind])
        func_message_list.append(four_list[3][ind])
        if four_list[0][ind] in dic_all:
            patch_list.append(dic_all[four_list[0][ind]])
        else:
            patch_list.append([four_list[0][ind].split('-')[0] + '-one'])

    name_number = [i.split('-')[0].strip() for i in name_number_func_list]

    re_name_number_func_list = []
    re_error_title_list = []
    re_error_message_list = []
    re_func_message_list = []
    re_patch_list = []

    remove_list_1 = ['Cli_6', 'Closure_63', 'Closure_93', 'Lang_2', 'Time_21']
    remove_list_2 = ['Collections_' + str(i) for i in range(1, 25)]
    remove_list = remove_list_1 + remove_list_2

    for i in range(len(name_number)):
        if name_number[i] not in remove_list:
            re_name_number_func_list.append(name_number_func_list[i])
            re_error_title_list.append(error_title_list[i])
            re_error_message_list.append(error_message_list[i])
            re_func_message_list.append(func_message_list[i])
            re_patch_list.append(patch_list[i])

    res = [re_name_number_func_list, re_error_title_list, re_error_message_list, re_func_message_list, re_patch_list]

    output = open('../data/test_case_all_five.pkl', 'wb')

    pickle.dump(res, output)


if __name__ == '__main__':
    get_link_patch()
