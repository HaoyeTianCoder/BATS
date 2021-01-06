import os
import shutil

path = '/Users/haoye.tian/Downloads/ODS/data/PS/'
target = '/Users/haoye.tian/Documents/University/data/PatchSimISSTA/PatchSim/'
data_139 = ['Patch1','Patch2','Patch4','Patch5','Patch6','Patch7','Patch8','Patch9','Patch10','Patch11','Patch12','Patch13','Patch14','Patch15','Patch16','Patch17','Patch18','Patch19','Patch20','Patch21','Patch22','Patch23','Patch24','Patch25','Patch26','Patch27','Patch28','Patch29','Patch30','Patch31','Patch32','Patch33','Patch34','Patch36','Patch37','Patch38','Patch44','Patch45','Patch46','Patch47','Patch48','Patch49','Patch51','Patch53','Patch54','Patch55','Patch58','Patch59','Patch62','Patch63','Patch64','Patch65','Patch66','Patch67','Patch68','Patch69','Patch72','Patch73','Patch74','Patch75','Patch76','Patch77','Patch78','Patch79','Patch80','Patch81','Patch82','Patch83','Patch84','Patch88','Patch89','Patch90','Patch91','Patch92','Patch93','Patch150','Patch151','Patch152','Patch153','Patch154','Patch155','Patch157','Patch158','Patch159','Patch160','Patch161','Patch162','Patch163','Patch165','Patch166','Patch167','Patch168','Patch169','Patch170','Patch171','Patch172','Patch173','Patch174','Patch175','Patch176','Patch177','Patch180','Patch181','Patch182','Patch183','Patch184','Patch185','Patch186','Patch187','Patch188','Patch189','Patch191','Patch192','Patch193','Patch194','Patch195','Patch196','Patch197','Patch198','Patch199','Patch201','Patch202','Patch203','Patch204','Patch205','Patch206','Patch207','Patch208','Patch209','Patch210','PatchHDRepair1','PatchHDRepair3','PatchHDRepair4','PatchHDRepair5','PatchHDRepair6','PatchHDRepair7','PatchHDRepair8','PatchHDRepair9','PatchHDRepair10']


def format(path, target):
    cnt = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                patchName = root.split('/')[-3]
                if patchName not in data_139:
                    continue

                label = root.split('/')[-4]
                project = root.split('/')[-2].split('_')[0]
                id = root.split('/')[-2].split('_')[1]

                patchid = 'patch1'

                path_target = os.path.join(target, label, project, id, patchid)
                if not os.path.exists(path_target):
                    os.makedirs(path_target)
                else:
                    number = len(os.listdir(os.path.join(target, label, project, id)))
                    patchid = 'patch' + str(int(number) + 1)
                    path_target = os.path.join(target, label, project, id, patchid)
                    os.makedirs(path_target)

                new_name = '-'.join([patchid, project, id, 'PatchSim']) + '.patch'
                shutil.copy(os.path.join(root, file), os.path.join(path_target, new_name))

                cnt += 1
                print(cnt)

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


format(path, target)
slice_patch2(target)