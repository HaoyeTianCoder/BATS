import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '.')))
# import seaborn as sns
import pickle
from experiment.config import Config
from representation.word2vector import Word2vector
import numpy as np
from scipy.spatial import distance
from experiment import patch_bert_vector
from experiment.evaluate import evaluation
from experiment.cluster import cluster


class Experiment:
    def __init__(self, path_test, path_patch_root, path_generated_patch, organized_dataset, patch_w2v):
        self.path_test = path_test
        # self.path_test_vector = path_test_vector
        # self.path_patch_vector = path_patch_vector
        self.path_patch_root = path_patch_root
        self.path_generated_patch = path_generated_patch

        self.organized_dataset = organized_dataset
        self.patch_w2v = patch_w2v

        self.original_dataset = None
        # self.patch_data = None

        self.test_name = None
        self.patch_name = None
        self.test_vector = None
        self.patch_vector = None
        self.exception_type = None

    def load_test(self,):
        # load original data
        with open(self.path_test, 'rb') as f:
            self.original_dataset = pickle.load(f)

        # organize data and vector
        if os.path.exists(organized_dataset):
            datasets = pickle.load(open(self.organized_dataset, 'rb'))
            self.test_name = datasets[0]
            self.patch_name = datasets[1]
            self.test_vector = datasets[2]
            self.patch_vector = datasets[3]
            self.exception_type = datasets[4]
        else:
            # learn the representation of test case and patch. always use code2vec for test case. 
            all_test_name, all_patch_name, all_test_vector, all_patch_vector, all_exception_type = self.test_patch_2vector(test_w2v='code2vec', patch_w2v=self.patch_w2v)
            
            # save data with different types to pickle.
            datasets = [all_test_name, all_patch_name, all_test_vector, all_patch_vector, all_exception_type]
            pickle.dump(datasets, open(self.organized_dataset, 'wb'))

            self.test_name = datasets[0]
            self.patch_name = datasets[1]
            self.test_vector = datasets[2]
            self.patch_vector = datasets[3]
            self.exception_type = datasets[4]


    def test_patch_2vector(self, test_w2v='code2vec', patch_w2v='cc2vec'):
        all_test_name, all_patch_name, all_test_vector, all_patch_vector, all_exception_type = [], [], [], [], []
        w2v = Word2vector(test_w2v=test_w2v, patch_w2v=patch_w2v, path_patch_root=self.path_patch_root)

        test_name_list = self.original_dataset[0]
        exception_type_list = self.original_dataset[1]
        log_list = self.original_dataset[2]
        test_function_list = self.original_dataset[3]
        # associated correct patch for failed test case. The name postfixing with '-one' means the complete patch hunk rather than part of it repaired the test case.
        patch_ids_list = self.original_dataset[4]
        for i in range(len(test_name_list)):
        # for i in tqdm(range(len(test_name_list))):
            name = test_name_list[i]
            function = test_function_list[i]
            ids = patch_ids_list[i]
            exception_type = exception_type_list[i]

            try:
                test_vector, patch_vector = w2v.convert_both(name, function, ids)
            except Exception as e:
                print('{} test name:{} exception emerge:{}'.format(i, name, e))
                continue
            print('{} test name:{} success!'.format(i, name,))

            all_test_name.append(name)
            all_patch_name.append(ids)
            all_test_vector.append(test_vector)
            all_patch_vector.append(patch_vector)
            all_exception_type.append(exception_type)

        if self.patch_w2v == 'string':
            return all_test_name, all_patch_name, np.array(all_test_vector), all_patch_vector, all_exception_type
        else:
            return all_test_name, all_patch_name, np.array(all_test_vector), np.array(all_patch_vector), all_exception_type

    def run(self, RQ):

        # load original data and corresponding vector
        self.load_test()

        # pre-save bert vector of patches if you use your own datasets(patches)
        # patch_bert_vector.patch_bert()

        # RQ1.1: validate hypothesis
        if 'RQ1.1' == RQ:
            clu = cluster(self.original_dataset, self.test_name, self.patch_name, self.test_vector, self.patch_vector, method='biKmeans', number=40)
            clu.validate()

        eval = evaluation(self.patch_w2v, self.original_dataset, self.test_name, self.test_vector, self.patch_vector, self.exception_type)
        # RQ1.2: evaluate on developer's patches of defects4j
        if 'RQ1.2' == RQ:
            eval.evaluate_defects4j_projects(option1=True, option2=0.6)

        # RQ2: evaluate BATS on the generated patches of APR tools. use cc2vec representation(patch_w2v='cc2vec') and cosine distance
        if 'RQ2' == RQ:
            # set cut-off = 0.0 to get baseline.
            eval.predict_collected_projects(path_collected_patch=self.path_generated_patch, cut_off=0.8, distance_method=distance.cosine, ASE2020=False, patchsim=False, )

        # RQ3.1: compare ML-based approach.
        if 'RQ3.1' == RQ:
            # ML-based approach.
            eval.predict_collected_projects(path_collected_patch=self.path_generated_patch, cut_off=0.8, distance_method=distance.cosine, ASE2020=True, patchsim=False )
            # patchsim: run experiment/evaluate_patchsim.py

        # RQ3.2: enhance PatchSim approach.
        if 'RQ3.2' == RQ:
            # ML-based approach. change patch_w2v to 'bert' in config.py as BATS with bert is better at -Recall.
            eval.predict_collected_projects(path_collected_patch=self.path_generated_patch, cut_off=0.6, distance_method=distance.euclidean, ASE2020=True, patchsim=False )
            # patchsim
            eval.predict_collected_projects(path_collected_patch=self.path_generated_patch, cut_off=0.6, distance_method=distance.cosine, ASE2020=False, patchsim=True )


if __name__ == '__main__':
    # specify RQ
    RQ = 'RQ2'

    config = Config()
    path_test = config.path_test
    path_patch_root = config.path_patch_root
    path_generated_patch = config.path_generated_patch
    organized_dataset = config.organized_dataset
    patch_w2v = config.patch_w2v

    e = Experiment(path_test, path_patch_root, path_generated_patch, organized_dataset, patch_w2v)
    e.run(RQ=RQ)