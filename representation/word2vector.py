from representation.code2vector import Code2vector
import pickle
from representation.CC2Vec import lmg_cc2ftr_interface

class Word2vector:
    def __init__(self, word2vec):
        self.w2v = word2vec

    def convert(self, data_text):
        # test_data: ['function A {...}','function B {}']
        if self.w2v == 'code2vec':
            test_vector = []
            for i in range(len(data_text)):
                test_function = data_text[i]
                vector = Code2vector().convert(test_function)
                test_vector.append(vector)
            return test_vector

        if self.w2v == 'cc2vec':
            dictionary = pickle.load(open('../CC2Vec/dict.pkl', 'rb'))
            patch_vector = []
            for i in range(len(data_text)):
                path_patch = data_text[i]
                learned_vector = lmg_cc2ftr_interface.learned_feature(path_patch, load_model='../CC2Vec/cc2ftr.pt', dictionary=dictionary)
                patch_vector.append(learned_vector)
            return patch_vector
