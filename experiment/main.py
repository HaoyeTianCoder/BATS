import pickle
from experiment.config import Config
from representation.word2vector import Word2vector
import numpy as np

class Experiment:
    def __init__(self, path_test, path_patch):
        self.path_test = path_test
        self.path_patch = path_patch

        self.test_data = None
        self.patch_data = None
        self.test_vector = None
        self.patch_vector = None

    def load_test(self,):
        self.test_data = pickle.load(self.path_test)
        self.patch_data = pickle.load(self.path_patch)

    def run(self):
        self.load_test()

        self.test2vector(word2v='code2vec')
        self.patch2vector(word2v='cc2vec')

        self.cal_all_simi(self.test_vector)

    def test2vector(self, word2v='code2vec'):
        w2v = Word2vector(word2v)
        self.test_vector = w2v.convert(self.test_data)

    def patch2vector(self, word2v='cc2vec'):
        w2v = Word2vector(word2v)
        self.patch_vector = w2v.convert(self.patch_data)

    def cal_all_simi(self, test_vector):
        test_vector = np.array(test_vector)
        center = np.mean(test_vector)
        dists = [np.linalg.norm(vec - center) for vec in test_vector]
        average = np.array(dists).mean()


if __name__ == '__main__':
    config = Config()
    path_test = config.path_test
    path_patch = config.path_patch

    e = Experiment(path_test, path_patch)
    e.run()