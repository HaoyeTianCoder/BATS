import json
from collect.configV2 import *
from collect.code_processing_all import get_trigger_path, get_error_path, get_name_func


def get_one_error(PATH_PROJECTS, NAME_LIST):
    """
    save case_one.json
    """

    trigger_path_list = get_trigger_path(PATH_PROJECTS, NAME_LIST)

    dic = {}

    for trigger_path in trigger_path_list:
        error_path_list = get_error_path(trigger_path)
        for error_path in error_path_list:

            f = open(error_path, 'r')
            lines = f.readlines()

            count = 0
            for line in lines:
                if '---' in line:
                    count += 1
            if count == 1:

                name = error_path.split('/')[-3]
                number = error_path.split('/')[-1]
                name_number = name + '_' + number
                name_func = get_name_func(error_path)[0]

                # name_number_func_list
                name_number_func = name_number+'-' + name_func

                dic[name_number_func] = [name_number + '-one']

    json.dump(dic, open(path_one_error, 'w'), sort_keys=True, indent=4)


if __name__ == '__main__':

    get_one_error(PATH_PROJECTS, NAME_LIST)