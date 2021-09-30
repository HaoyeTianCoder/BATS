import json
import os

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1Natural'
path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1Natural_sliced'
cnt = 0

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.patch'):
            project = root.split('/')[-5]
            patch_id = file.split('-')[0]
            patch_folder = root.split('/')[-1]
            if patch_folder != patch_id:
                new_path = '/'.join(root.split('/')[:-1]+[patch_id])
                try:
                    os.rename(root, new_path)
                except Exception as e:
                    print(e)
                cnt += 1
print(cnt)