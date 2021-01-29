import json
import os

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1ISSTA/'
cnt = 0
unique = set()
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.patch'):
            if root.split('/')[-3] == 'Mockito':
                continue
            cnt += 1
            key = ''
            try:
                with open(os.path.join(root, file), 'r+') as f:
                    for line in f:
                        if line.startswith('--') or line.startswith('++')  or line.startswith('diff') or line.startswith('index') \
                            or line.startswith('Index') or line.startswith('==='):
                            pass
                        else:
                            key += line
            except Exception as e:
                print(os.path.join(root, file))
                continue

            if key not in unique:
                unique.add(key)
            print('cnt: {}'.format(cnt))
print('cnt: {}, unique: {}'.format(cnt, len(unique)))