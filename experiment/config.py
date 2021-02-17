class Config:
    def __init__(self):
        # the orginal data of test case name, test function, associated patch including 'single test case' and 'full' versions.
        self.path_test = '../data/test_case_all.pkl'
        # self.path_test = '../data/test_one_case_all.pkl'

        # developers' patches in defects4j and generated patches of APR tools
        self.path_patch_root = '/Users/haoye.tian/Documents/University/project/defects4j_patch_sliced/'
        self.path_generated_patch = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1ISSTA_sliced/'

        # choose one type of representations to learn the behaviour of patch
        self.patch_w2v = 'cc2vec'
        # self.patch_w2v = 'bert'
        # self.patch_w2v = 'string'

        self.organized_dataset = '../data/organized_dataset_' + self.patch_w2v + '.pickle'
        self.path_test_function_patch_vector = '../data/vector_one_case_' + self.patch_w2v + '.pickle'


if __name__ == '__main__':
    Config()