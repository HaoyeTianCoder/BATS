import os
import shutil

path = '/Users/haoye.tian/Downloads/ODS/data/PS/'
target = '/Users/haoye.tian/Documents/University/data/PatchSimISSTA/'


def format(path, target):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                label = root.split('/')[-4]
                project = root.split('/')[-2].split('_')[0]
                id = root.split('/')[-2].split('_')[1]
                patchid = 'patch1'

                path_target = os.path.join(target, label, project, id, patchid)
                if not os.path.exists(path_target):
                    os.makedirs(path_target)

                new_name = '-'.join([patchid, project, id, 'PatchSim']) + '.patch'

                shutil.copy(os.path.join(root, file), os.path.join(path_target, new_name))

def slice_patch2(path):
    cnt = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                id = root.split('/')[-2]
                project = root.split('/')[-3]
                name = file.split('.')[0]
                number_diff = 0
                number_AT = 0
                patch = ''
                content = False
                try:
                    with open(os.path.join(root, file)) as f:
                        for line in f:
                            if line.startswith('--- ') or line.startswith('-- '):
                                if number_diff > 0:
                                    # save previous patch
                                    new_path = root.replace('PatchSimISSTA', 'PatchSimISSTA_sliced')
                                    new_name = name+'-'+str(number_AT)+'.patch'
                                    if not os.path.exists(new_path):
                                        os.makedirs(new_path)
                                    with open(os.path.join(new_path, new_name), 'w+') as f:
                                        f.write(minus_line + plus_line + patch)

                                    content = False
                                else:
                                    number_diff += 1
                                minus_line = line
                            elif line.startswith('+++ ') or line.startswith('++ '):
                                plus_line = line
                            elif line.startswith('@@ '):
                                if content:
                                    # save previous patch
                                    new_path = root.replace('PatchSimISSTA', 'PatchSimISSTA_sliced')
                                    new_name = name+'-'+str(number_AT)+'.patch'
                                    if not os.path.exists(new_path):
                                        os.makedirs(new_path)
                                    with open(os.path.join(new_path, new_name), 'w+') as f:
                                        f.write(minus_line + plus_line + patch)

                                    patch = line
                                    number_AT += 1
                                    content = True
                                else:
                                    # first @@
                                    patch = line
                                    number_AT += 1
                                    content = True
                            elif content:
                                patch += line
                            else:
                                continue
                except Exception as e:
                    print(e)

                # save last patch
                new_path = root.replace('PatchSimISSTA', 'PatchSimISSTA_sliced')
                new_name = name + '-' + str(number_AT) + '.patch'
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                with open(os.path.join(new_path, new_name), 'w+') as f:
                    f.write(minus_line + plus_line + patch)


# format(path, target)
slice_patch2(target)