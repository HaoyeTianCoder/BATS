class Config:
    def __init__(self):
        self.path_test = '../data/test_case_all.pkl'
        # self.path_test_vector = '../data/test_case_function.npy'

        # self.path_patch_root = '/Users/lyh/Documents/thy/defects4j_patch_sliced/'
        self.path_patch_root = '/Users/haoye.tian/Documents/University/project/defects4j_patch_sliced/'
        self.path_collected_patch = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1ISSTA_sliced/'

        self.path_test_function_patch_vector = '../data/vector_case_cc2vec.pickle'
        self.patch_w2v = 'cc2vec'

        # self.path_test_function_patch_vector = '../data/vector_case_bert.pickle'
        # self.patch_w2v = 'bert'

        # self.path_test_function_patch_vector = '../data/vector_case_str.pickle'
        # self.patch_w2v = 'str'
if __name__ == '__main__':
    Config()