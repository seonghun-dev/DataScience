import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('data_week12.csv')
X = np.array(df[df.columns.difference(['Class variable (0 or 1).'])])
y = np.array(df['Class variable (0 or 1).'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def classificationPerformanceEval(y, y_predict, name='Classifier'):
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

    print(f"{name} accuracy = %f " % accuracy)
    print(f"{name} precision = %f" % precision)
    print(f"{name} recall = %f " % recall)
    print(f"{name} f1_score = %f" % f1_score)
    return accuracy, precision, recall, f1_score


dtree = DecisionTreeClassifier()
dtree_model = dtree.fit(X_train, y_train)
y_pred = dtree_model.predict(X_test)
classificationPerformanceEval(y_test, y_pred, name='Decision Tree')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
classificationPerformanceEval(y_test, y_pred, name='Logistic Regression')
