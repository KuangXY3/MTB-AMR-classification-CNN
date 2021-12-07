"""select most important feature sets for the models of the 4 drugs,
by trying different feature_imp_threshold.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Here, the order of features in feat_labels should be same as the order of the features in feature matrix 'featureM_X_drug.txt'
feat_labels = np.loadtxt("raw_fList.txt", dtype=np.str)
# print ("Number of full set of features: {}".format(len(feat_labels))  )
# first line drugs
drug_l = ["rifampicin", "isoniazid", "pyrazinamide", "ethambutol"]
# second line drugs
# drug_l = ['amikacin','capreomycin','kanamycin','ofloxacin']
for drug in drug_l:
    featureX = "featureM_X_" + drug + ".txt"
    label = "label_Y_" + drug + ".txt"
    X = np.loadtxt(featureX, dtype="i4")
    y = np.loadtxt(label, dtype="i4")
    f = 0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0
    )
    clf = RandomForestClassifier(
        n_estimators=1000, random_state=0, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    feature_imp_threshold = min(clf.feature_importances_)
    ma = max(clf.feature_importances_)
    # print ('The range of feature importances: {}-{}'.format(min(clf.feature_importances_),max(clf.feature_importances_)))
    # for feature in zip(feat_labels, clf.feature_importances_):
    #    print(feature)

    # Apply The Full Featured Classifier To The Test Data
    y_pred = clf.predict(X_test)

    # View The Accuracy Of Our Full Feature set (283 Features) Model
    a_fullF = accuracy_score(y_test, y_pred)
    print("Using full set of features on drug {}".format(drug))
    # print(a_fullF)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    Precision = tp / float(tp + fp)
    # print ('Precision:'+str(Precision))
    Recall = tp / float(tp + fn)
    print("F-Measure:" + str(2 * (Recall * Precision) / (Recall + Precision)))
    # print(tn, fp, fn, tp)

    print("Using selected feature sets by iterating feature importance threshold")
    # find the best feature_imp_threshold
    while feature_imp_threshold < ma:
        sfm = SelectFromModel(clf, threshold=feature_imp_threshold)
        sfm.fit(X_train, y_train)

        # Transform the data to create a new dataset containing only the most important features
        # Note: We have to apply the transform to both the training X and test X data.
        X_important_train = sfm.transform(X_train)
        X_important_test = sfm.transform(X_test)

        # Create a new random forest classifier for the most important features
        clf_important = RandomForestClassifier(
            n_estimators=1000, class_weight="balanced", random_state=0, n_jobs=-1
        )

        # Train the new classifier on the new dataset containing the most important features
        clf_important.fit(X_important_train, y_train)

        # Apply The selected Featured Classifier To The Test Data

        y_important_pred = clf_important.predict(X_important_test)
        # View The Accuracy Of Our Limited Feature  Model
        # a_selectedF=accuracy_score(y_test, y_important_pred)

        # print('Feature importance threshold: {}'.format(feature_imp_threshold))
        # print(a_selectedF)
        tn, fp, fn, tp = confusion_matrix(y_test, y_important_pred).ravel()
        Precision = tp / float(tp + fp)
        # print ('Precision:'+str(Precision))
        Recall = tp / float(tp + fn)
        f_measure = 2 * (Recall * Precision) / (Recall + Precision)
        print("{},{}".format(feature_imp_threshold, f_measure))
        # print ('F-Measure:'+str(f_measure))
        # print(tn, fp, fn, tp)
        if f_measure > f:
            f = f_measure
            best_thr = feature_imp_threshold
            best_selected_model = sfm

        feature_imp_threshold += 0.0001
    print("Best f-measure: {}; best importance threashold: {}".format(f, best_thr))
    for feature_list_index in best_selected_model.get_support(indices=True):
        print(feat_labels[feature_list_index])
