import os
import shutil
from subprocess import *

path_defects4j = '/Users/haoye.tian/Documents/University/project/defects4j/framework/projects/'
defects4j_patch = '/Users/haoye.tian/Documents/University/project/defects4j_patch2/'
projects = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']


def obtain_patchV1():
    cnt = 0
    success = 0
    for project in projects:
        for root, dirs, files in os.walk(path_defects4j+project+ '/patches'):
            for file in files:
                if file.endswith('.src.patch'):
                    cnt += 1
                    id = file.split('.')[0]
                    patch = ''
                    try:
                        with open(os.path.join(root, file), 'r+', newline='\r\n') as f:
                            for line in f:
                                if line.startswith('-') and not line.startswith('---'):
                                    patch += '+' + line[1:]
                                elif line.startswith('+') and not line.startswith('+++'):
                                    patch += '-' + line[1:]
                                else:
                                    patch += line
                    except Exception as e:
                        print('name:{}'.format(project+' '+id))

                    # write into patch
                    new_path = defects4j_patch + project
                    new_name = id+'.patch'
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    with open(os.path.join(new_path, new_name), 'w+') as f:
                        f.write(patch)
                    success += 1

                    print('success/cnt: {}/{}'.format(success, cnt))

path_defects4j_buggy = '/Users/haoye.tian/Documents/University/project/defects4j_buggy/'
def obtain_patchV2(path_defects4j_buggy):
    bug_ids = os.listdir(path_defects4j_buggy)
    for bug_id in bug_ids:
        if bug_id.startswith('.'):
            continue
        bug = bug_id.split('_')[0]
        id = bug_id.split('_')[1]
        path_bug_id = path_defects4j_buggy + bug_id
        buggy_version = 'D4J_' + bug_id + '_BUGGY_VERSION'
        fixed_version = 'D4J_' + bug_id + '_FIXED_VERSION'
        new_path = os.path.join(defects4j_patch, bug, id, 'patch1')
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        cmd = "cd {} && git diff {} {} -- . ':!.*' >  {}".format(path_bug_id, buggy_version, fixed_version, os.path.join(new_path, 'patch1-'+bug+'-'+id+'-Developer.patch'))
        try:
            with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                output, errors = p.communicate(timeout=300)
                # print(output)
                if errors:
                    raise CalledProcessError(errors, '-1')
        except Exception as e:
            print(e)

obtain_patchV2(path_defects4j_buggy)