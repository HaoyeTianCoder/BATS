import os
import json
import pickle

from collect.configV2 import *


def get_trigger_path(path_projects, name_list):
    """
    return path list
    """
    res = []
    for name in name_list:
        tmp = path_projects + name + '/trigger_tests'
        res.append(tmp)
    return res


def get_error_path(trigger_path):
    """
    return error path list
    """
    error_path_list = []
    for file in os.listdir(trigger_path):
        file_path = os.path.join(trigger_path, file)
        error_path_list.append(file_path)
    return error_path_list


def count_error_number(trigger_path_list):
    # Get all error cases
    count_value = [0] * 10000
    key = list(range(10000))
    dic = dict(zip(key, count_value))
    for trigger_path in trigger_path_list:
        error_path_list = get_error_path(trigger_path)
        for error_path in error_path_list:

            f = open(error_path, 'r')
            lines = f.readlines()
            count = 0
            for line in lines:
                if '---' in line:
                    count += 1
            dic[count] += 1

    new_dict = {key: val for key, val in dic.items() if val != 0}
    json.dump(new_dict, open('test_case_numberV2.json', 'w'), sort_keys=True, indent=4)


def get_buggy_path(error_path):
    f = open(error_path, 'r')
    lines = f.readlines()
    for line in lines:
        if '---' in line:
            path = line.split('::')
            buggy_path = path[0].split('---')[-1].strip().replace('.', '/')
            case_func = path[-1]
            break
    return buggy_path, case_func


def get_error_title(error_path):
    f = open(error_path, 'r')
    lines = f.readlines()
    for ind in range(len(lines)):
        if '---' in lines[ind]:
            ind_title = ind+1
            break

    return lines[ind_title].strip()


def get_name_func(error_path):
    f = open(error_path, 'r')
    lines = f.readlines()
    for ind in range(len(lines)):
        if '---' in lines[ind]:
            break

    line = lines[ind]
    line = line.split('::')
    return line[-1].strip()


def get_error_message(error_path):
    f = open(error_path, 'r')
    lines = f.readlines()

    ind_start = 0
    ind_end = len(lines)

    flag = 0
    for ind in range(len(lines)):
        if '---' in lines[ind]:
            if flag == 0:
                ind_start = ind+2
                flag += 1
            else:
                ind_end = ind
                break
    error_message_list = lines[ind_start: ind_end]
    error_message_list = [error_message.strip() for error_message in error_message_list]
    return error_message_list[::-1]


def get_func_list(error_path, buggy_path):

    func = get_name_func(error_path)
    try:
        f = open(buggy_path, 'r', encoding="ISO-8859-1")
    except:
        path = buggy_path.replace('/java/', '/')
        try:
            f = open(path, 'r', encoding="ISO-8859-1")
        except:
            path = buggy_path.replace('/test/', '/test/java/')
            f = open(path, 'r', encoding="ISO-8859-1")

    lines = f.readlines()
    ind_start = 0
    ind_end = 0

    for ind in range(len(lines)):
        if 'void ' + func + '()' in lines[ind]:
            ind_start = ind
            break

    left = 0
    flag = 0
    for ind in range(ind_start, len(lines)):
        for s in lines[ind]:
            if s == '{':
                left += 1
            elif s == '}':
                left -= 1
                if left == 0:
                    ind_end = ind
                    flag = 1
                    break
        if flag:
            break

    func_message = lines[ind_start: ind_end+1]
    res = ''
    for line in func_message:
        res += line
    if ind_start:
        return res
    else:
        return '-1'


def get_all(PATH_PROJECTS, NAME_LIST):
    trigger_path_list = get_trigger_path(PATH_PROJECTS, NAME_LIST)

    name_number_list = []
    error_title_list = []
    error_message_embed_list = []
    case_func_list = []

    for trigger_path in trigger_path_list:
        error_path_list = get_error_path(trigger_path)
        for error_path in error_path_list:
            try:

                name = error_path.split('/')[-3]
                number = error_path.split('/')[-1]
                name_number = name + '_' + number

                buggy_path = dic_buggy_path[name].replace(name, name_number)
                code_buggy_path, case_func = get_buggy_path(error_path)
                buggy_path = buggy_path + code_buggy_path + '.java'

                error_title = get_error_title(error_path)
                error_message_list = get_error_message(error_path)
                func_mesage = get_func_list(error_path, buggy_path)

                if func_mesage != '-1':
                    name_number_list.append(name_number)
                    error_title_list.append(error_title)
                    error_message_embed_list.append(error_message_list)
                    case_func_list.append(func_mesage)

            except:
                pass


    print(len(name_number_list))
    print(len(error_title_list))
    print(len(error_message_embed_list))
    print(len(case_func_list))
    #
    # print(name_number_list[25])
    # print(error_title_list[25])
    # print(error_message_embed_list[25])
    # print(case_func_list[345])

    res = [name_number_list, error_title_list, error_message_embed_list, case_func_list]

    output = open('../data/test_caseV2.pkl', 'wb')
    pickle.dump(res, output)

    trigger_path_list = get_trigger_path(PATH_PROJECTS, NAME_LIST)
    count_error_number(trigger_path_list)


def main():
    get_all(PATH_PROJECTS, NAME_LIST)


if __name__ == '__main__':
    main()
