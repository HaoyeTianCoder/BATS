from .code2vector import Code2vector

class Word2vector:
    def __init__(self, word2vec):
        self.w2v = word2vec

    def convert(self, test_data):
        # test_data: ['function A {...}','function B {}']
        if self.w2v == 'code2vec':
            test_vector = []
            for i in range(len(test_data)):
                test_function = test_data[i]
                vector = Code2vector().convert(test_function)
                test_vector.append(vector)
            return test_vector

        if self.w2v == 'cc2vec':
            pass
