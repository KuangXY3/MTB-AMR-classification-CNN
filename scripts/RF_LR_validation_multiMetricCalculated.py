import numpy as np

# from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer

drugL = ["ethambutol", "isoniazid", "pyrazinamide", "rifampicin"]
# import feature labels
feat_labels = np.loadtxt("raw_fList.txt", dtype=np.str)


def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 0]


def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 1]


def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])[1, 0]


def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])[1, 1]


scoring = {
    "tp": make_scorer(tp),
    "tn": make_scorer(tn),
    "fp": make_scorer(fp),
    "fn": make_scorer(fn),
}


for drug in drugL:
    print(drug)
    X = np.loadtxt("featureM_X_" + drug + ".txt", dtype="i4")
    y = np.loadtxt("label_Y_" + drug + ".txt", dtype="i4")
    # clf = RandomForestClassifier(n_estimators=1000 ,random_state=0, n_jobs=-1, class_weight="balanced")
    clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    rf_results = cross_validate(clf.fit(X, y), X, y, scoring=scoring, cv=10)
    log = LogisticRegression(n_jobs=-1, penalty="l2")
    # log = LogisticRegression(n_jobs=-1,penalty='l1', random_state=0, class_weight="balanced")
    lr_results = cross_validate(log.fit(X, y), X, y, scoring=scoring, cv=10)

    re_models = {"rf": rf_results, "lr": lr_results}

    for key, value in re_models.items():
        print("Evaluation of " + key)
        s_tn = sum(value["test_tn"])
        s_tp = sum(value["test_tp"])
        s_fn = sum(value["test_fn"])
        s_fp = sum(value["test_fp"])

        print("tp:", s_tp, ",", "tn:", s_tn, ",", "fp:", s_fp, ",", "fn:", s_fn)
        print("Accuracy:" + str((s_tp + s_tn) / float(s_tp + s_tn + s_fp + s_fn)))
        # print ('Recall:'+str(s_tp/float(s_tp+s_fn)))
        print("Specificity:" + str(s_tn / float(s_tn + s_fp)))
        print("Sensitivity:" + str(s_tp / float(s_tp + s_fn)))
        Precision = s_tp / float(s_tp + s_fp)
        print("Precision:" + str(s_tp / float(s_tp + s_fp)))
        Recall = s_tp / float(s_tp + s_fn)
        print("F-Measure:" + str(2 * (Recall * Precision) / (Recall + Precision)))
        # print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
