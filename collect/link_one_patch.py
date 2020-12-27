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

    dic_case_one = get_data_json(path_one_error)

    name_number_func_list = []
    error_title_list = []
    error_message_list = []
    func_message_list = []
    patch_list = []

    for ind in range(len(four_list[0])):
        if four_list[0][ind] in dic_case_one:
            patch_list.append(dic_case_one[four_list[0][ind]])
            name_number_func_list.append(four_list[0][ind])
            error_title_list.append(four_list[1][ind])
            error_message_list.append(four_list[2][ind])
            func_message_list.append(four_list[3][ind])

    name_number = [i.split('-')[0].strip() for i in name_number_func_list]

    re_name_number_func_list = []
    re_error_title_list = []
    re_error_message_list = []
    re_func_message_list = []
    re_patch_list = []

    # remove Deprecated bug ids
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

    # remove 922
    if 'Lang_57-org.apache.commons.lang.LocaleUtilsTest::testCountriesByLanguage' in re_name_number_func_list:
        ind = re_name_number_func_list.index('Lang_57-org.apache.commons.lang.LocaleUtilsTest::testCountriesByLanguage')
        re_name_number_func_list.pop(ind)
        re_error_title_list.pop(ind)
        re_error_message_list.pop(ind)
        re_func_message_list.pop(ind)
        re_patch_list.pop(ind)

    res = [re_name_number_func_list, re_error_title_list, re_error_message_list, re_func_message_list, re_patch_list]

    output = open('../data/test_one_case_all.pkl', 'wb')

    pickle.dump(res, output)

    # print(re_name_number_func_list[200])
    # print(re_func_message_list[200])
    # print(re_patch_list[200])


if __name__ == '__main__':
    get_link_patch()
