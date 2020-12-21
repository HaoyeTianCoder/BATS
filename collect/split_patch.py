import os

path = '/Users/haoye.tian/Documents/University/project/defects4j_patch/'

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
                project = root.split('/')[-1]
                name = file.split('.')[0]
                id = name
                number_diff = 0
                number_AT = 0
                patch = ''
                content = False
                try:
                    with open(os.path.join(root, file)) as f:
                        for line in f:
                            if line.startswith('--- '):
                                minus_line = line
                            elif line.startswith('+++ '):
                                plus_line = line
                            elif line.startswith('diff '):
                                if number_diff > 0:
                                    # save previous patch
                                    new_path = path.replace('defects4j_patch', 'defects4j_patch_sliced') + project + '/' + id
                                    new_name = id+'_'+str(number_AT)+'.patch'
                                    if not os.path.exists(new_path):
                                        os.makedirs(new_path)
                                    with open(os.path.join(new_path, new_name), 'w+') as f:
                                        f.write(minus_line + plus_line + patch)

                                    content = False
                                else:
                                    number_diff += 1
                                    continue
                            elif line.startswith('@@ '):
                                if content:
                                    # save previous patch
                                    new_path = path.replace('defects4j_patch', 'defects4j_patch_sliced') + project + '/' + id
                                    new_name = id+'_'+str(number_AT)+'.patch'
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
                new_path = path.replace('defects4j_patch', 'defects4j_patch_sliced') + project + '/' + id
                new_name = id + '_' + str(number_AT) + '.patch'
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                with open(os.path.join(new_path, new_name), 'w+') as f:
                    f.write(minus_line + plus_line + patch)

# slice_patch(path)
slice_patch2(path)