import numpy as np
import pymysql
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


# 호출한 지표들을 한꺼번에 계산하는 함수 정의
def classificationPerformanceEval(y_test, y_predict):
    print(classification_report(y_test, y_predict, target_names=['A', 'B', 'C']))


def SVC(x, y):
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    from sklearn.svm import SVC

    svc = SVC()
    svc.fit(x_train, y_train)
    y_predict = svc.predict(x_test)

    classificationPerformanceEval(y_test, y_predict)

    # kfold cross validation
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    np_y_predict = []
    np_y_test = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svc = SVC(random_state=0)
        svc.fit(x_train, y_train)
        y_predict = svc.predict(x_test)
        for index, value in enumerate(list(y_test)):
            np_y_test.append(value)
        for index, value in enumerate(list(y_predict)):
            np_y_predict.append(value)
    classificationPerformanceEval(np.array(np_y_test), np.array(np_y_predict))


def logisticRegression(x, y):
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=90)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    lr_model = lr.fit(x_train, y_train)
    y_predict_proba = lr_model.predict_proba(x_test)
    from sklearn.preprocessing import Binarizer
    custom_threshold = 0.5
    y_predict = y_predict_proba[:, 1].reshape(-1, 1)
    binarizer = Binarizer(threshold=custom_threshold).fit(y_predict)
    custom_y_predict = binarizer.transform(y_predict)

    classificationPerformanceEval(y_test, custom_y_predict)

    # kfold cross validation
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=2, random_state=42, shuffle=True)
    np_y_predict = []
    np_y_test = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lr = LogisticRegression(max_iter=1000)
        lr_model = lr.fit(x_train, y_train)
        y_predict_proba = lr_model.predict_proba(x_test)
        from sklearn.preprocessing import Binarizer
        custom_threshold = 0.3
        y_predict = y_predict_proba[:, 1].reshape(-1, 1)
        binarizer = Binarizer(threshold=custom_threshold).fit(y_predict)
        custom_y_predict = binarizer.transform(y_predict)

        for index, value in enumerate(list(y_test)):
            np_y_test.append(value)
        for index, value in enumerate(list(custom_y_predict)):
            np_y_predict.append(value)
    classificationPerformanceEval(np.array(np_y_test), np.array(np_y_predict))


conn = pymysql.connect(host="localhost", user="root", password="6674", db='dataSceince')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = "select * from db_score_3"
curs.execute(sql)

data = curs.fetchall()

curs.close()
conn.close()

x = [(t['sno'], t['homework'], t['final'], t['discussion']) for t in data]
x = np.array(x)

y = []
for t in data:
    if t['grade'] == 'A':
        y.append(0)
    elif t['grade'] == 'B':
        y.append(1)
    else:
        y.append(2)
y = np.array(y)

SVC(x, y)
logisticRegression(x, y)
