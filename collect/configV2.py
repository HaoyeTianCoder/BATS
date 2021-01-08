PATH_PROJECTS = '/Users/lyh/Documents/thy/similarity/V2/defects4j/framework/projects/'


# NAME_LIST = ['Chart', 'Closure', 'JacksonDatabind', 'Jsoup', 'Lang', 'Math', 'Mockito', 'Time',
#              'Cli', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore', 'JacksonXml', 'JxPath']

NAME_LIST = ['Chart', 'JacksonDatabind', 'Jsoup', 'Lang', 'Math', 'Mockito', 'Time',
             'Cli', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore', 'JacksonXml', 'JxPath']

user_path = '/Users/lyh/Documents/thy/similarity/V2/'

dic_buggy_path = {
    'Chart': user_path+'defects4j_buggy/Chart/tests/',
    # 'Closure': user_path+'defects4j_buggy/Closure/test/',
    'JacksonDatabind': user_path+'defects4j_buggy/JacksonDatabind/src/test/java/',
    'Jsoup': user_path+'defects4j_buggy/Jsoup/src/test/java/',
    'Lang': user_path+'defects4j_buggy/Lang/src/test/java/',
    'Math': user_path+'defects4j_buggy/Math/src/test/java/',
    'Mockito': user_path+'defects4j_buggy/Mockito/test/',
    'Time': user_path+'defects4j_buggy/Time/src/test/java/',

    'Cli': user_path+'defects4j_buggy/Cli/src/test/',
    'Codec': user_path+'defects4j_buggy/Codec/src/test/',
    'Collections': user_path+'defects4j_buggy/Collections/src/test/java/',
    'Compress': user_path+'defects4j_buggy/Compress/src/test/java/',
    'Csv': user_path+'defects4j_buggy/Csv/src/test/java/',
    'Gson': user_path+'defects4j_buggy/Gson/gson/src/test/java/',
    'JacksonCore': user_path+'defects4j_buggy/JacksonCore/src/test/java/',
    'JacksonXml': user_path+'defects4j_buggy/JacksonXml/src/test/java/',
    'JxPath': user_path+'defects4j_buggy/JxPath/src/test/'
}


path_haoye = 'case_patch_haoye.json'
path_yinghua = 'case_patch_yinghua.json'
path_weiguo = 'case_patch_weiguo.json'

path_one_error = 'case_one.json'

missing_name_number = ['Closure_93', 'Closure_63', 'Lang_2', 'Time_21', 'Cli_6',
                       'Collections_20', 'Collections_18', 'Collections_9', 'Collections_11',
                       'Collections_7', 'Collections_16', 'Collections_6', 'Collections_17',
                       'Collections_1', 'Collections_10', 'Collections_19', 'Collections_8',
                       'Collections_21', 'Collections_24', 'Collections_23', 'Collections_4',
                       'Collections_15', 'Collections_3', 'Collections_12', 'Collections_2',
                       'Collections_5', 'Collections_14', 'Collections_22']