import json
import os

# path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1Natural'
path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1Natural_sliced'
cnt = 0
cnt2 = 0

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.patch'):
            name = file.split('.')[0]
            tool = root.split('/')[-5]
            patch_id = file.split('-')[0]
            patch_folder = root.split('/')[-1]
            if patch_folder != patch_id:
                new_path = '/'.join(root.split('/')[:-1]+[patch_id])
                try:
                    os.rename(root, new_path)
                except Exception as e:
                    print(e)
                cnt += 1

            tool_name = file.split('.')[0].split('-')[3]
            if tool_name != tool:
                new_patch_name = name.replace(tool_name, tool)
                os.rename(os.path.join(root,name+'.patch'), os.path.join(root,new_patch_name)+'.patch')
                cnt2 += 1

                if os.path.exists(os.path.join(root, name+'.txt')):
                    os.rename(os.path.join(root, name+'.txt'), os.path.join(root, new_patch_name) + '.txt')

print(cnt)
print(cnt2)