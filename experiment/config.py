class Config:
    def __init__(self):
        self.path_test = '../data/test_case_all_five.pkl'
        # self.path_test_vector = '../data/test_case_function.npy'

        self.path_patch_root = '/Users/haoye.tian/Documents/University/project/defects4j_patch_sliced/'
        # self.path_patch_vector = '../data/test_case_function_.npy'

        self.path_test_function_patch_vector = '../data/test_function_patch_vector.pickle'


if __name__ == '__main__':
    Config()