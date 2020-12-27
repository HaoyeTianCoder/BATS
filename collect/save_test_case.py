import os
import json
from collect.configV2 import *
from collect.code_processing_all import get_buggy_path


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
    dic_name_number = {}

    remove_list_1 = ['Cli_6', 'Closure_63', 'Closure_93', 'Lang_2', 'Time_21']
    remove_list_2 = ['Collections_' + str(i) for i in range(1, 25)]
    remove_list = remove_list_1 + remove_list_2

    for trigger_path in trigger_path_list:
        error_path_list = get_error_path(trigger_path)
        for error_path in error_path_list:

            name = error_path.split('/')[-3]
            number = error_path.split('/')[-1]
            name_number = name + '_' + number

            if name_number not in missing_name_number and name_number not in remove_list:

                f = open(error_path, 'r')
                lines = f.readlines()
                count = 0
                for line in lines:
                    if '---' in line:
                        count += 1

                dic[count] += 1

                dic_name_number[name_number] = []
                for ind in range(len(lines)):
                    if '---' in lines[ind]:
                        line = lines[ind]
                        line = line.split('::')
                        dic_name_number[name_number].append(line[-1].strip())

    # Math_41-org.apache.commons.math.stat.descriptive.moment.VarianceTest::testEvaluateArraySegmentWeighted, no function
    new_dict = {key: val for key, val in dic.items() if val != 0}
    json.dump(new_dict, open('test_case_numberV2.json', 'w'), sort_keys=True, indent=4)


    new_dic_name_number = {key: val for key, val in dic_name_number.items() if len(val) > 1}
    json.dump(new_dic_name_number, open('test_case_name_numberV2.json', 'w'), sort_keys=True, indent=4)

if __name__ == '__main__':
    trigger_path_list = get_trigger_path(PATH_PROJECTS, NAME_LIST)
    count_error_number(trigger_path_list)

    # remove Deprecated bug ids
