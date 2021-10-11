import numpy as np
import pymysql


def classificationPerformanceEval(y, y_predict):
    tp, tn, fp, fn = 0, 0, 0, 0

    for y, yp in zip(y, y_predict):
        if y == 1 and yp == 1:
            tp += 1
        elif y == 1 and yp == -1:
            fn += 1
        elif y == -1 and yp == 1:
            fp += 1
        else:
            tn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1_score


def decisionTree(x, y):
    # train data, test data 분류 작업
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    from sklearn import tree

    dtree = tree.DecisionTreeClassifier()

    dtree_model = dtree.fit(x_train, y_train)

    y_predict = dtree_model.predict(x_test)

    acc, prec, rec, f1 = classificationPerformanceEval(y_test, y_predict)

    print("Decision Tree accuracy = %f " % acc)
    print("Decision Tree precision = %f" % prec)
    print("Decision Tree recall = %f " % rec)
    print("Decision Tree f1_score = %f" % f1)

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

        dtree = tree.DecisionTreeClassifier()
        dtree_model = dtree.fit(x_train, y_train)
        y_predict = dtree_model.predict(x_test)
        acc, prec, rec, f1 = classificationPerformanceEval(y_test, y_predict)

        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)

    import statistics

    print("average accuracy = ", statistics.mean(accuracy))
    print("average precision = ", statistics.mean(precision))
    print("average recall = ", statistics.mean(recall))
    print("average f1_score = ", statistics.mean(f1_score))


def SVC(x, y):
    # train data, test data 분류 작업
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    from sklearn.svm import SVC

    svc = SVC(random_state=0)
    svc.fit(x_train, y_train)
    y_predict = svc.predict(x_test)

    acc, prec, rec, f1 = classificationPerformanceEval(y_test, y_predict)

    print("LogisticRegression accuracy = %f " % acc)
    print("LogisticRegression precision = %f" % prec)
    print("LogisticRegression recall = %f " % rec)
    print("LogisticRegression f1_score = %f" % f1)


conn = pymysql.connect(host="localhost", user="root", password="6674", db='dataSceince')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = "select * from db_score_3"
curs.execute(sql)

data = curs.fetchall()

curs.close()
conn.close()

x = [(t['sno'], t['homework'], t['discussion'], t['final']) for t in data]
x = np.array(x)

y = [1 if (t['grade'] == 'B') else -1 for t in data]
y = np.array(y)

decisionTree(x, y)

SVC(x,y)
