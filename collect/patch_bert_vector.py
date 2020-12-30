import os
import shutil
import json
import pickle
from representation.word2vector import Word2vector

path_patch_sliced = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1ISSTA_sliced/'

def patch_bert_vector():
    w2v = Word2vector(patch_w2v='bert', )
    projects = {'Chart': 26, 'Lang': 65, 'Time': 27, 'Closure': 176, 'Math': 106}
    # projects = {'Time': 2,}
    # with open('../data/patch_vector.pickle', 'w+') as f:
    for project, number in projects.items():
        print('Berting {}'.format(project))
        for id in range(1, number + 1):
            tools = os.listdir(path_patch_sliced)
            for label in ['Correct', 'Incorrect']:
                for tool in tools:
                    path_bugid = os.path.join(path_patch_sliced, tool, label, project, str(id))
                    if os.path.exists(path_bugid):
                        patches = os.listdir(path_bugid)
                        for p in patches:
                            path_patch = os.path.join(path_bugid, p)
                            # json_key = '-'.join([path_patch.split('/')[-5], path_patch.split('/')[-4], path_patch.split('/')[-3], path_patch.split('/')[-2], path_patch.split('/')[-1]])
                            if not os.path.isdir(path_patch):
                                continue
                            vector = w2v.convert_single_patch(path_patch)
                            vector_list = list(vector)
                            vector_list = list(map(str, vector_list))

                            json_key = path_patch + '_.json'
                            print('json_key: {}'.format(json_key))
                            with open(json_key, 'w+') as f:
                                jsonstr = json.dumps(vector_list, )
                                f.write(jsonstr)
        # pickle.dump(dict, f)
        # f.write(jsonstr)

patch_bert_vector()