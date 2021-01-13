from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, confusion_matrix

class MlPrediction:
    def __init__(self, x_train, y_train, x_test, y_test, algorithm='lr'):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.algorithm = algorithm

    def predict(self):
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()

        x_train, y_train = self.x_train, self.y_train
        x_test, y_test = self.x_test, self.y_test

        # standard data
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        print('The number of train: {}, The number of test: {}'.format(len(x_train), len(x_test)))

        clf = None
        if self.algorithm == 'lr':
            clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train, y=y_train)
        elif self.algorithm == 'dt':
            clf = DecisionTreeClassifier().fit(X=x_train, y=y_train, sample_weight=None)
        elif self.algorithm == 'rf':
            clf = RandomForestClassifier(n_estimators=100, ).fit(X=x_train, y=y_train)

        y_pred = clf.predict_proba(x_test)[:, 1]

        auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)

    def evaluation_metrics(self, y_true, y_pred_prob):
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
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