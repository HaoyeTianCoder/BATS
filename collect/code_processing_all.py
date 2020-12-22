import os
import json
import pickle

from collect.configV2 import *


def get_trigger_path(path_projects, name_list):
    """
    return path list.
    """
    res = []
    for name in name_list:
        tmp = path_projects + name + '/trigger_tests'
        res.append(tmp)
    return res


def get_error_path(trigger_path):
    """
    return error path list.
    """
    error_path_list = []
    for file in os.listdir(trigger_path):
        file_path = os.path.join(trigger_path, file)
        error_path_list.append(file_path)
    return error_path_list


def count_error_number(trigger_path_list):
    """
    save test case number json.
    """
    count_value = [0] * 10000
    key = list(range(10000))
    dic = dict(zip(key, count_value))
    dic_name_number = {}
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

            name = error_path.split('/')[-3]
            number = error_path.split('/')[-1]
            name_number = name + '_' + number
            dic_name_number[name_number] = []

            for ind in range(len(lines)):
                if '---' in lines[ind]:
                    line = lines[ind]
                    line = line.split('::')
                    dic_name_number[name_number].append(line[-1].strip())


    new_dict = {key: val for key, val in dic.items() if val != 0}
    json.dump(new_dict, open('test_case_numberV2.json', 'w'), sort_keys=True, indent=4)

    new_dic_name_number = {key: val for key, val in dic_name_number.items() if len(val) > 1}
    json.dump(new_dic_name_number, open('test_case_name_numberV2.json', 'w'), sort_keys=True, indent=4)



def get_buggy_path(error_path):
    """
    return list buggy path.
    """
    error_path_list = []
    f = open(error_path, 'r')
    lines = f.readlines()
    for line in lines:
        if '---' in line:
            path = line.split('::')
            buggy_path = path[0].split('---')[-1].strip().replace('.', '/')
            error_path_list.append(buggy_path)
    return error_path_list


def get_error_title(error_path):
    """
    return error title list.
    """
    error_title_list = []
    f = open(error_path, 'r')
    lines = f.readlines()
    for ind in range(len(lines)):
        if '---' in lines[ind]:
            ind_title = ind+1
            title = lines[ind_title].strip()
            error_title_list.append(title)

    return error_title_list


def get_name_func(error_path):
    """
    return name func list.
    """
    name_func_list = []
    f = open(error_path, 'r')
    lines = f.readlines()
    for ind in range(len(lines)):
        if '---' in lines[ind]:
            line = lines[ind]
            func = line.split('---')[-1].strip()
            name_func_list.append(func)

    return name_func_list

def get_single_func(name_func_list):
    """
    return single func list.
    """
    name_single_func = []
    for func_char in name_func_list:
        single_func = func_char.split('::')[-1].strip()
        name_single_func.append(single_func)
    return name_single_func


def get_error_message(error_path):
    """
    return error message list.
    """
    error_message_list = []
    f = open(error_path, 'r')
    lines = f.readlines()

    ind_start = 0
    ind_list = [i for i in range(len(lines)) if '---' in lines[i]]
    if len(ind_list) == 1:
        ind_start = ind_start + 2
        error_message = lines[ind_start:]
        error_message_ = [message.strip() for message in error_message]
        error_message = error_message_[::-1]
        error_message_list.append(error_message)
        return error_message_list
    elif len(ind_list) > 1:
        for i in range(1, len(ind_list)):
            ind_start = ind_start + 2
            ind_end = ind_list[i]
            error_message = lines[ind_start: ind_end]
            error_message_ = [message.strip() for message in error_message]
            error_message = error_message_[::-1]
            error_message_list.append(error_message)
            ind_start = ind_end
        ind_start = ind_start+2
        error_message = lines[ind_start:]
        error_message_ = [message.strip() for message in error_message]
        error_message = error_message_[::-1]
        error_message_list.append(error_message)

        return error_message_list


def get_func_message(func, buggy_path):
    """
    return func message.
    """
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
            if s == '{' and '"{" around' not in lines[ind] and "'{';" not in lines[ind]:  # Closure_173, JacksonCore_25 special handling
                left += 1
            elif s == '}':
                left -= 1
                if left == 0:
                    ind_end = ind
                    flag = 1
                    break
        if flag:
            break

    func_message = lines[ind_start: ind_end + 1]
    res = ''
    for line in func_message:
        res += line
    if ind_start:
        return res
    else:
        return '-1'


def get_all(PATH_PROJECTS, NAME_LIST):
    trigger_path_list = get_trigger_path(PATH_PROJECTS, NAME_LIST)

    re_name_number_func_list = []
    re_error_title_list = []
    re_error_message_list = []
    re_func_message_list = []

    for trigger_path in trigger_path_list:
        error_path_list = get_error_path(trigger_path)
        for error_path in error_path_list:
            try:

                name = error_path.split('/')[-3]
                number = error_path.split('/')[-1]

                name_number = name + '_' + number
                name_func_list = get_name_func(error_path)
                name_single_func_list = get_single_func(name_func_list)

                # name_number_func_list
                name_number_func_list = [name_number+'-'+i for i in name_func_list]

                buggy_path = dic_buggy_path[name].replace(name, name_number)
                code_buggy_path_list = get_buggy_path(error_path)
                buggy_path_list = []
                for path in code_buggy_path_list:
                    tmp_buggy_path = buggy_path + path + '.java'
                    buggy_path_list.append(tmp_buggy_path)

                # error_title_list
                error_title_list = get_error_title(error_path)

                # error_message_list
                error_message_list = get_error_message(error_path)

                # func_message_list
                func_message_list = []
                for i in range(len(buggy_path_list)):
                    tmp_func = name_single_func_list[i]
                    tmp_buggy_path = buggy_path_list[i]
                    tmp_func_message = get_func_message(tmp_func, tmp_buggy_path)
                    func_message_list.append(tmp_func_message)

                re_name_number_func_list += name_number_func_list
                re_error_title_list += error_title_list
                re_error_message_list += error_message_list
                re_func_message_list += func_message_list

            except:
                pass


    filter_re_name_number_func_list = []
    filter_re_error_title_list = []
    filter_re_error_message_list = []
    filter_re_func_message_list = []

    for ind in range(len(re_func_message_list)):
        if re_func_message_list[ind] != '-1':
            filter_re_name_number_func_list.append(re_name_number_func_list[ind])
            filter_re_error_title_list.append(re_error_title_list[ind])
            filter_re_error_message_list.append(re_error_message_list[ind])
            filter_re_func_message_list.append(re_func_message_list[ind])

    print(len(filter_re_name_number_func_list))
    print(len(filter_re_error_title_list))
    print(len(filter_re_error_message_list))
    print(len(filter_re_func_message_list))

    # print(filter_re_name_number_func_list[110])
    # print(filter_re_error_title_list[110])
    # print(filter_re_error_message_list[110])
    # print(filter_re_func_message_list[110])

    res = [filter_re_name_number_func_list, filter_re_error_title_list, filter_re_error_message_list, filter_re_func_message_list]

    output = open('../data/test_case_all.pkl', 'wb')
    pickle.dump(res, output)

    return res

def main():

    get_all(PATH_PROJECTS, NAME_LIST)


if __name__ == '__main__':
    main()
