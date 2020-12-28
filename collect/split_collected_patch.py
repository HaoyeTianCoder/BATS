import os
import shutil


def slice_patch(path):
    cnt = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]
                number_diff = 0
                patch = ''
                try:
                    with open(os.path.join(root, file)) as f:

                        for line in f:
                            if line.startswith('diff --git') or line.startswith('Index:'):
                                if number_diff == 0:
                                    patch += line
                                    number_diff += 1
                                else:
                                    # save previous patch
                                    new_path = root.replace('defects4j_patch', 'defects4j_patch_sliced')
                                    if not os.path.exists(new_path):
                                        os.makedirs(new_path)
                                    with open(os.path.join(new_path, name +'_'+str(number_diff)+ '.patch'), 'w+') as f:
                                        f.write(patch)

                                    # reset for new one
                                    patch = line
                                    number_diff += 1
                            else:
                                patch += line

                        # handle last hunk
                        new_path = root.replace('defects4j_patch', 'defects4j_patch_sliced')
                        if not os.path.exists(new_path):
                            os.makedirs(new_path)
                        with open(os.path.join(new_path, name + '_' + str(number_diff) + '.patch'), 'w+') as f:
                            f.write(patch)
                except Exception as e:
                    print(name)

                if number_diff > 1:
                    cnt += 1
                    print('{} patches with multiple fixes found'.format(cnt))

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
                                    new_path = root.replace('PatchCollectingV1ISSTA', 'PatchCollectingV1ISSTA_sliced')
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
                                    new_path = root.replace('PatchCollectingV1ISSTA', 'PatchCollectingV1ISSTA_sliced')
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
                new_path = root.replace('PatchCollectingV1ISSTA', 'PatchCollectingV1ISSTA_sliced')
                new_name = name + '-' + str(number_AT) + '.patch'
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                with open(os.path.join(new_path, new_name), 'w+') as f:
                    f.write(minus_line + plus_line + patch)

def combine_patch(path):
    cnt = 0
    for root, dirs, files in os.walk(path):
        if len(files) > 1 and files[0].endswith('.patch'):
            frag = files[0].split('-')
            new_name = '-'.join([frag[0].split('_')[0], frag[1], frag[2], frag[3]])
            new_str_patch = ''
            for file in files:
                with open(os.path.join(root, file), 'r+') as f:
                    new_str_patch += ''.join(f.readlines())
            for file in files:
                os.remove(os.path.join(root, file))

            # save new patch
            with open(os.path.join(root, new_name), 'w+') as f:
                f.write(new_str_patch)

def move(path):
    cnt = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                if '_' in file:
                    patchid = file.split('_')[0]
                else:
                    patchid = file.split('-')[0]
                new_root = root.replace('PatchCollectingV1', 'PatchCollectingV1ISSTA') + '/'+patchid
                if not os.path.exists(new_root):
                    os.makedirs(new_root)
                shutil.copy(os.path.join(root, file), os.path.join(new_root, file))

# move('/Users/haoye.tian/Documents/University/data/PatchCollectingV1/')
path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1ISSTA/'
# combine_patch(path)
slice_patch2(path)