import os

path = '/Users/haoye.tian/Documents/University/project/defects4j_patch/'

def slice_patch(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]
                try:
                    with open(os.path.join(root, file)) as f:
                        number_diff = 0
                        patch = ''
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

slice_patch(path)