import numpy as np
import pymysql


def classificationPerformanceEval(y, y_predict):
    tp, tn, fp, fn = 0, 0, 0, 0

    for y, yp in zip(y, y_predict):
        if y == 1 and yp == 1:
            tp += 1
        elif y == 1 and yp == 0:
            fn += 1
        elif y == 0 and yp == 1:
            fp += 1
        else:
            tn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1_score


def SVC(x, y):
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    from sklearn.svm import SVC

    svc = SVC()
    svc.fit(x_train, y_train)
    y_predict = svc.predict(x_test)

    acc, prec, rec, f1 = classificationPerformanceEval(y_test, y_predict)

    print("-----------SVM train_test_split-----------")
    print("SVM accuracy = %f " % acc)
    print("SVM precision = %f" % prec)
    print("SVM recall = %f " % rec)
    print("SVM f1_score = %f" % f1)

    # kfold cross validation
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    accuracy = []
    precision = []
    recall = []
    f1_score = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svc = SVC(random_state=0)
        svc.fit(x_train, y_train)
        y_predict = svc.predict(x_test)
        acc, prec, rec, f1 = classificationPerformanceEval(y_test, y_predict)

        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)

    import statistics
    print("-----------SVM K-fold cross validation-----------")
    print("SVM average accuracy = ", statistics.mean(accuracy))
    print("SVM average precision = ", statistics.mean(precision))
    print("SVM average recall = ", statistics.mean(recall))
    print("SVM average f1_score = ", statistics.mean(f1_score))


def logisticRegression(x, y):
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=90)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr_model = lr.fit(x_train, y_train)
    y_predict_proba = lr_model.predict_proba(x_test)
    from sklearn.preprocessing import Binarizer
    custom_threshold = 0.5
    y_predict = y_predict_proba[:, 1].reshape(-1, 1)
    binarizer = Binarizer(threshold=custom_threshold).fit(y_predict)
    custom_y_predict = binarizer.transform(y_predict)

    acc, prec, rec, f1 = classificationPerformanceEval(y_test, custom_y_predict)
    print("-----------logistic Regression train_test_split-----------")
    print("logistic Regression accuracy = %f " % acc)
    print("logistic Regression precision = %f" % prec)
    print("logistic Regression recall = %f " % rec)
    print("logistic Regression f1_score = %f" % f1)

    # kfold cross validation
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=2, random_state=42, shuffle=True)
    accuracy = []
    precision = []
    recall = []
    f1_score = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lr = LogisticRegression()
        lr_model = lr.fit(x_train, y_train)
        y_predict_proba = lr_model.predict_proba(x_test)
        from sklearn.preprocessing import Binarizer
        custom_threshold = 0.3
        y_predict = y_predict_proba[:, 1].reshape(-1, 1)
        binarizer = Binarizer(threshold=custom_threshold).fit(y_predict)
        custom_y_predict = binarizer.transform(y_predict)

        acc, prec, rec, f1 = classificationPerformanceEval(y_test, custom_y_predict)

        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)

    import statistics
    print("-----------logistic Regression cross validation-----------")
    print("logistic average accuracy = ", statistics.mean(accuracy))
    print("logistic average precision = ", statistics.mean(precision))
    print("logistic average recall = ", statistics.mean(recall))
    print("logistic average f1_score = ", statistics.mean(f1_score))


conn = pymysql.connect(host="localhost", user="root", password="6674", db='dataSceince')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = "select * from db_score_3"
curs.execute(sql)

data = curs.fetchall()

curs.close()
conn.close()

x = [(t['sno'], t['homework'], t['final'], t['discussion']) for t in data]
x = np.array(x)

y = [1 if (t['grade'] == 'B') else 0 for t in data]
y = np.array(y)

SVC(x, y)
logisticRegression(x, y)
