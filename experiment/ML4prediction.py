from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

class MlPrediction:
    def __init__(self, x_train, y_train, x_test, y_test, y_pred_bats, test_case_similarity_list, algorithm='lr', comparison=True, cutoff=0.8):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred_bats = y_pred_bats
        self.test_case_similarity_list = test_case_similarity_list

        self.algorithm = algorithm
        self.comparison = comparison
        self.cutoff = cutoff

    def confusion_matrix(self, y_pred, y_test):
        for i in range(1, 100):
            y_pred_tn = [1 if p >= i / 100.0 else 0 for p in y_pred]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tn).ravel()
            print('i:{}'.format(i / 100), end=' ')
            # print('TP: %d -- TN: %d -- FP: %d -- FN: %d' % (tp, tn, fp, fn))
            recall_p = tp / (tp + fn)
            recall_n = tn / (tn + fp)
            print('+Recall: {:.3f}, -Recall: {:.3f}'.format(recall_p, recall_n))

    def predict(self, comparison=False):
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()

        x_train, y_train = self.x_train, self.y_train
        x_test, y_test = self.x_test, self.y_test

        # standard data
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        # # deduplicate, deprecated
        # x_test_unique = [list(x_test[i]) for i in range(len(x_test))]
        # x_train_unique = []
        # for i in range(len(x_train)):
        #     if list(x_train[i]) in x_test_unique:
        #         continue
        #     else:
        #         x_train_unique.append(x_train[i])

        print('train data: {}, test data: {}'.format(len(x_train), len(x_test)))

        clf = None
        if self.algorithm == 'lr':
            clf = LogisticRegression().fit(X=x_train, y=y_train)
        elif self.algorithm == 'dt':
            clf = DecisionTreeClassifier().fit(X=x_train, y=y_train, sample_weight=None)
        elif self.algorithm == 'rf':
            clf = RandomForestClassifier().fit(X=x_train, y=y_train)
        elif self.algorithm == 'nb':
            clf = GaussianNB().fit(X=x_train, y=y_train)

        y_pred = clf.predict_proba(x_test)[:, 1]
        print('{}: '.format(self.algorithm))

        print('1. ML:')
        auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)
        # self.confusion_matrix(y_pred, y_test)

        # # combine ML-based and BATs
        if self.cutoff == 0.6:
            print('2. Combine(replace):')
            y_pred_prob_combine = []
            BATs, ML = 0, 0
            for i in range(len(self.y_pred_bats)):
                if self.test_case_similarity_list[i] >= 0.9:
                    y_pred_prob_combine.append(self.y_pred_bats[i])
                    BATs += 1
                else:
                    y_pred_ML = clf.predict_proba(x_test[i].reshape(1, -1))[:, 1]
                    y_pred_prob_combine.append(y_pred_ML)
                    ML += 1
            print('BATs:{}, ML:{}'.format(BATs, ML))
            auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=y_test, y_pred_prob=y_pred_prob_combine)
            # self.confusion_matrix(y_pred_prob_combine, y_test)

        # average ML-based and BATs, deprecated
        # print('3. Combine(average):')
        # y_pred = (y_pred + self.y_pred_bats)/2.0
        # auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)
        # self.confusion_matrix(y_pred, y_test)

    def evaluation_metrics(self, y_true, y_pred_prob):
        threshold = 0.5
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
        auc_ = auc(fpr, tpr)

        if self.comparison and self.cutoff == 0.8:
            if self.algorithm == 'lr':
                threshold = 0.07
            elif self.algorithm == 'rf':
                threshold = 0.37

        y_pred = [1 if p >= threshold else 0 for p in y_pred_prob]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        prc = precision_score(y_true=y_true, y_pred=y_pred)
        rc = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = 2 * prc * rc / (prc + rc)

        print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
        # return , auc_
        return auc_, recall_p, recall_n, acc, prc, rc, f1