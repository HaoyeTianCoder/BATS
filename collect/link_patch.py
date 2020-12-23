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
        if four_list[0][ind] in dic_all:
            name_number_func_list.append(four_list[0][ind])
            error_title_list.append(four_list[1][ind])
            error_message_list.append(four_list[2][ind])
            func_message_list.append(four_list[3][ind])
            patch_list.append(dic_all[four_list[0][ind]])

    res = [name_number_func_list, error_title_list, error_message_list, func_message_list, patch_list]

    output = open('../data/test_case_all_five.pkl', 'wb')

    pickle.dump(res, output)
    print(len(patch_list))


if __name__ == '__main__':
    get_link_patch()
